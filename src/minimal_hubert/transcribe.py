from pathlib import Path

import orjson
import torch
from filelock import FileLock
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

from .features import split_for_distributed


def transcribe(root: str | Path, kmeans: MiniBatchKMeans, jsonl: str | Path) -> None:
    root = Path(root)
    files = split_for_distributed(sorted(root.rglob("*.pt")))
    lock = FileLock(f"{jsonl}.lock")
    for path in tqdm(files):
        fileid = str(path.relative_to(root)).removesuffix(".pt")
        units = kmeans.predict(torch.load(path).numpy()).tolist()
        units_str = orjson.dumps({"file": fileid, "units": units}, option=orjson.OPT_APPEND_NEWLINE)
        with lock, Path(jsonl).open("ab") as f:
            f.write(units_str)


if __name__ == "__main__":
    import argparse

    import joblib

    parser = argparse.ArgumentParser(description="Inference of discrete units")
    parser.add_argument("root_features", type=Path, help="Root directory containing feature files")
    parser.add_argument("path_kmeans", type=Path, help="Path to the trained K-means model")
    parser.add_argument("output_jsonl", type=Path, help="Path to the output JSONL file with units")
    args = parser.parse_args()
    transcribe(args.root_features, joblib.load(args.path_kmeans), args.output_jsonl)
