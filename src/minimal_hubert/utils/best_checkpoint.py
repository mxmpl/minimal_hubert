import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger()


def find_and_symlink_best_checkpoint(checkpoint: str | Path, validation: str | Path) -> None:
    df = pl.read_ndjson(validation).sort("name")
    best = df[df["loss"].arg_min()].to_dicts()[0]
    logger.info("Best checkpoint: '%s' with %s val. loss", best["name"], best["loss"])
    (Path(checkpoint) / "best.pt").symlink_to(f"{best['name']}.pt")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("path_checkpoints", type=Path, help="Directory containing all intermediate checkpoints")
    parser.add_argument("path_validation", type=Path, help="Path to the JSONL file with validation losses")
    args = parser.parse_args()
    find_and_symlink_best_checkpoint(args.path_checkpoints, args.path_validation)
