from collections.abc import Iterable
from pathlib import Path
from tempfile import TemporaryDirectory

import orjson
import torch
from fastabx import zerospeech_abx
from filelock import FileLock
from spidr.config import SAMPLE_RATE
from torch.nn import functional as F
from torchcodec.decoders import AudioDecoder
from tqdm import tqdm

from .model import HuBERTPretrain
from .utils import slurm_job_tmpdir


def extract_features(
    model: HuBERTPretrain,
    audio: Path,
    features: Path,
    layers: set[int],
    *,
    extension: str,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    num_layers = max(layers)
    for path in tqdm(sorted(audio.rglob(f"*{extension}")), desc="Extract features"):
        samples = AudioDecoder(path).get_all_samples()
        assert samples.sample_rate == SAMPLE_RATE
        data = F.layer_norm(samples.data.to(device), samples.data.shape)
        for i, feats in enumerate(model.get_intermediate_outputs(data, num_layers=num_layers)):
            if i + 1 not in layers:
                continue
            dest = features / str(i + 1) / path.with_suffix(".pt").name
            dest.parent.mkdir(exist_ok=True, parents=True)
            torch.save(feats.squeeze().cpu(), dest)


def compute_and_save_abx(
    item: str | Path,
    audio: str | Path,
    output: str | Path,
    *,
    checkpoint: str | Path,
    layers: Iterable[int] | None = None,
    extension: str = ".wav",
    tmpdir: str | Path | None = None,
) -> None:
    item_name = Path(item).stem
    model = HuBERTPretrain.from_pretrained(checkpoint)
    layers = set(range(1, 13) if layers is None else layers)
    assert min(layers) >= 1
    assert max(layers) <= 12
    lock = FileLock(f"{output}.lock")
    with TemporaryDirectory(dir=tmpdir) as features:
        extract_features(model, Path(audio), Path(features), layers, extension=extension)
        for layer in layers:
            this_features = Path(features) / str(layer)
            for speaker in ("within", "across"):
                score = zerospeech_abx(item, this_features, max_size_group=10, max_x_across=5, speaker=speaker)
                result = {"item": item_name, "layer": layer, "speaker": speaker, "score": score}
                with lock, Path(output).open("ab") as f:
                    f.write(orjson.dumps(result, option=orjson.OPT_APPEND_NEWLINE))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute ABX discriminability of HuBERT representations")
    parser.add_argument("item", type=Path, help="Path to the item file")
    parser.add_argument("audio", type=Path, help="Path to the audio files")
    parser.add_argument("output", type=Path, help="Path to the output JSONL file with ABX scores")
    parser.add_argument("--checkpoint", required=True, help="Path to the HuBERT checkpoint file")
    parser.add_argument("--layers", nargs="+", type=int, help="Target layers")
    parser.add_argument("--extension", type=str, default=".wav", help="Audio file extension")
    parser.add_argument(
        "--tmpdir",
        type=Path,
        help="Path to the temporary directory where features will be stored",
        default=slurm_job_tmpdir(),
    )
    args = parser.parse_args()

    compute_and_save_abx(
        args.item,
        args.audio,
        args.output,
        checkpoint=args.checkpoint,
        layers=args.layers,
        extension=args.extension,
        tmpdir=args.tmpdir,
    )
