from os.path import commonpath
from pathlib import Path

import torch
from spidr.data import speech_dataset
from torch import Tensor
from torchaudio.compliance import kaldi
from torchaudio.functional import compute_deltas
from tqdm import tqdm

from .model import HuBERTPretrain
from .utils import split_for_distributed


def mfcc(waveform: Tensor) -> Tensor:
    mfccs = kaldi.mfcc(waveform).transpose(0, 1)
    deltas = compute_deltas(mfccs)
    ddeltas = compute_deltas(deltas)
    return torch.cat([mfccs, deltas, ddeltas], dim=0).transpose(0, 1).contiguous()


def compute_and_save_mfccs(path_manifest: str | Path, root_features: str | Path) -> None:
    dataset = speech_dataset(path_manifest, normalize=False)
    root = commonpath(dataset.manifest["path"])
    indices = split_for_distributed(list(range(len(dataset))))
    dest = Path(root_features)
    for i in tqdm(indices):
        name = Path(dataset.manifest[i, "path"]).relative_to(root).with_suffix(".pt")
        waveform = dataset[i].unsqueeze(0)
        features = mfcc(waveform)
        (dest / name.parent).mkdir(exist_ok=True, parents=True)
        torch.save(features, dest / name)


@torch.no_grad()
def compute_and_save_hubert_features(
    path_manifest: str | Path,
    root_features: str | Path,
    checkpoint: str | Path,
    layer: int,
) -> None:
    dataset = speech_dataset(path_manifest, normalize=True)
    root = commonpath(dataset.manifest["path"])
    indices = split_for_distributed(list(range(len(dataset))))
    dest = Path(root_features)
    model = HuBERTPretrain.from_pretrained(checkpoint).cuda()
    for i in tqdm(indices):
        name = Path(dataset.manifest[i, "path"]).relative_to(root).with_suffix(".pt")
        waveform = dataset[i].unsqueeze(0).cuda()
        features = model.get_intermediate_outputs(waveform, num_layers=layer)[-1].squeeze().cpu()
        (dest / name.parent).mkdir(exist_ok=True, parents=True)
        torch.save(features, dest / name)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract features using MFCC or HuBERT")
    parser.add_argument("path-manifest", type=Path, help="Path to the manifest file")
    parser.add_argument("root-features", type=Path, help="Root directory for output features")
    parser.add_argument("--type", required=True, choices=["mfcc", "hubert"], help="Feature type: 'mfcc' or 'hubert'")
    parser.add_argument("--path-checkpoint", type=Path, help="Path to HuBERT checkpoint file")
    parser.add_argument("--layer", type=int, help="Layer number to extract features from")
    args = parser.parse_args()
    if args.type == "mfcc":
        compute_and_save_mfccs(args.path_manifest, args.root_features)
    else:
        compute_and_save_hubert_features(args.path_manifest, args.root_features, args.path_checkpoint, args.layer)
