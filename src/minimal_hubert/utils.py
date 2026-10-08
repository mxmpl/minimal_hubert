import logging
import math
import os
import socket
from collections.abc import Sequence
from pathlib import Path

import polars as pl
from spidr.config import DEFAULT_CONV_LAYER_CONFIG
from spidr.data.utils import read_manifest

logger = logging.getLogger()


def split_for_distributed[T](sequence: Sequence[T]) -> Sequence[T]:
    if "SLURM_NTASKS" not in os.environ:
        return sequence
    rank, world_size = int(os.environ["SLURM_PROCID"]), int(os.environ["SLURM_NTASKS"])
    array_id, num_arrays = int(os.getenv("SLURM_ARRAY_TASK_ID", "0")), int(os.getenv("SLURM_ARRAY_TASK_COUNT", "1"))
    if "SLURM_ARRAY_TASK_ID" in os.environ:
        assert os.environ["SLURM_ARRAY_TASK_MIN"] == "0"
        assert int(os.environ["SLURM_ARRAY_TASK_MAX"]) == num_arrays - 1

    n_total = len(sequence)  # Split by array first
    files_per_array = math.ceil(n_total / num_arrays)
    start = array_id * files_per_array
    end = min(start + files_per_array, n_total)
    sequence = sequence[start:end]

    n_local = len(sequence)  # Then split by rank within each array
    files_per_rank = math.ceil(n_local / world_size)
    start = rank * files_per_rank
    end = min(start + files_per_rank, n_local)
    return sequence[start:end]


def slurm_job_tmpdir() -> Path | None:
    if "JOBSCRATCH" in os.environ:
        return Path(os.environ["JOBSCRATCH"])
    if (path := Path(f"/fastscratch/{socket.gethostname()}")).is_dir() and "SLURM_JOB_ID" in os.environ:
        return path / os.environ["SLURM_JOB_ID"]
    return None


def conv_length_expr(num_samples: pl.Expr) -> pl.Expr:  # spidr.data.dataset.conv_length, as a polars expression
    for _, kernel_size, stride in DEFAULT_CONV_LAYER_CONFIG:
        num_samples = ((num_samples - kernel_size) // stride + 1).clip(lower_bound=0)
    return num_samples


def scan_manifest(path: str | Path) -> pl.LazyFrame:
    match Path(path).suffix:
        case ".csv":
            return pl.scan_csv(path)
        case ".jsonl":
            return pl.scan_ndjson(path)
    return read_manifest(path).lazy()


def merge_manifest_with_units(path_manifest: str, path_units: str, output: str, *, from_mfcc: bool) -> None:
    # The units file can be very large: it is only streamed, never fully loaded in memory.
    # Rows are written in the order of the units file, and files without units are dropped.
    manifest = scan_manifest(path_manifest)
    units = pl.scan_ndjson(path_units, schema={"fileid": pl.String, "units": pl.List(pl.Int32)})

    # First pass on the fileids only, to check that the join is one-to-one
    manifest_ids = manifest.select("fileid").collect()["fileid"]
    units_ids = units.select("fileid").collect(engine="streaming")["fileid"]
    for name, ids in (("manifest", manifest_ids), ("units", units_ids)):
        if ids.is_duplicated().any():
            msg = f"Duplicate fileids in the {name} file, e.g. {ids.filter(ids.is_duplicated())[0]!r}"
            raise ValueError(msg)
    if num_missing := int((~manifest_ids.is_in(units_ids.implode())).sum()):
        logger.warning("%d/%d files in the manifest have no units and are dropped", num_missing, len(manifest_ids))
    del manifest_ids, units_ids

    if from_mfcc:  # MFCC frames are every 10ms, HuBERT frames every 20ms
        units = units.with_columns(pl.col("units").list.gather_every(2))
    columns = manifest.collect_schema().names()
    (  # Units are streamed through, only the (small) manifest is held in memory for the join
        units.join(manifest, on="fileid", how="inner", build_side="prefer_right", maintain_order="left")
        .select(*columns, pl.col("units").list.head(conv_length_expr(pl.col("num_samples"))))
        .sink_ndjson(output, engine="streaming")
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", help="Path to manifest file")
    parser.add_argument("units", help="Path to the JSONL file with units")
    parser.add_argument("output", help="Path to the output manifest file with units")
    parser.add_argument(
        "--from-mfcc",
        action="store_true",
        help="Add this flag if units are derived from MFCC (10ms instead of 20ms)",
    )
    args = parser.parse_args()
    merge_manifest_with_units(args.manifest, args.units, args.output, from_mfcc=args.from_mfcc)
