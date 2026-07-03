import math
import os
import socket
from collections.abc import Sequence
from pathlib import Path

import polars as pl
from spidr.config import DEFAULT_CONV_LAYER_CONFIG
from spidr.data.dataset import conv_length


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


def merge_manifest_with_units(path_manifest: str, path_units: str, *, from_mfcc: bool) -> pl.DataFrame:
    fileids = set(
        pl.scan_ndjson(path_manifest)
        .select("fileid")
        .join(pl.scan_ndjson(path_units).select("fileid"), on="fileid", validate="1:1")
        .sort("fileid")
        .collect()
        .to_series()
    )
    length = conv_length(DEFAULT_CONV_LAYER_CONFIG, pl.read_ndjson(path_manifest)["num_samples"].to_torch())
    return (
        pl.concat(
            (
                pl.read_ndjson(path_manifest).sort("fileid"),
                pl.scan_ndjson(path_units)
                .sort("fileid")
                .filter(pl.col("fileid").is_in(fileids))
                .drop("fileid")
                .with_columns(pl.col("units").list.gather_every(2) if from_mfcc else pl.col("units"))
                .collect(),
            ),
            how="horizontal",
        )
        .with_columns(pl.Series(name="length", values=length))
        .with_columns(pl.col("units").list.slice(offset=0, length=pl.col("length")))
        .drop("length")
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
    merge_manifest_with_units(args.manifest, args.units, from_mfcc=args.from_mfcc).write_ndjson(args.output)
