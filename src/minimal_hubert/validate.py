import logging
from pathlib import Path

import orjson
import polars as pl
import torch
from spidr.config import MaskingConfig
from spidr.environment import set_seed, setup_environment, setup_pytorch
from spidr.tools import init_logger
from torch.utils.data import DataLoader

from .config import hubert_data_config
from .data import build_dataloader_with_labels
from .model import HuBERT

logger = logging.getLogger()


@torch.no_grad()
def validate(model: HuBERT, loader: DataLoader, device: torch.device, dtype: torch.dtype) -> dict[str, float]:
    model.eval()
    total_loss = torch.zeros(1, device=device)
    total_feature_loss = torch.zeros(1, device=device)
    mixed_precision = dtype != torch.float32
    for waveforms, labels, attention_mask, mask in loader:
        with torch.autocast("cuda", dtype, mixed_precision):
            loss, outputs = model(
                waveforms.to(device),
                labels.to(device),
                mask=mask.to(device),
                attention_mask=attention_mask.to(device),
            )
        total_loss += loss.mean()
        total_feature_loss += outputs["feature_loss"]
    total_loss /= len(loader)
    total_feature_loss /= len(loader)
    return {"loss": total_loss.item(), "feature_loss": total_feature_loss.item()}


def validate_all_checkpoints(manifest: str, checkpoints: str | Path, output: str | Path, *, seed: int = 0) -> None:
    init_logger()
    set_seed(seed)
    setup_pytorch(use_deterministic=False)
    setup_environment()
    device, dtype = torch.device("cuda"), torch.bfloat16
    loader = build_dataloader_with_labels(hubert_data_config(manifest), MaskingConfig())
    for path in sorted(Path(checkpoints).glob("*.pt")):
        model = HuBERT.from_pretrained(path).to(device)
        losses = validate(model, loader, device, dtype)
        with Path(output).open("ab") as f:
            f.write(orjson.dumps({"name": path.stem} | losses, option=orjson.OPT_APPEND_NEWLINE))


def find_and_symlink_best_checkpoint(checkpoints: str | Path, validation: str | Path) -> None:
    best = pl.read_ndjson(validation).sort("name").filter(pl.col("loss") == pl.col("loss").min()).to_dicts()[0]
    logger.info("Best checkpoint: '%s' with %s val. loss", best["name"], best["loss"])
    (Path(checkpoints) / "best.pt").symlink_to(f"{best['name']}.pt")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run validation for all existing HuBERT checkpoints")
    parser.add_argument("path_manifest", type=str, help="Path to the validation manifest file")
    parser.add_argument("path_checkpoints", type=Path, help="Directory containing all intermediate checkpoints")
    parser.add_argument("output_validation", type=Path, help="Path to the output JSONL file with validation losses")
    args = parser.parse_args()
    validate_all_checkpoints(args.path_manifest, args.path_checkpoints, args.output_validation)
    find_and_symlink_best_checkpoint(args.path_checkpoints, args.output_validation)
