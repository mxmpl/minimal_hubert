from pathlib import Path

import orjson
import torch
from filelock import FileLock
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

from .utils import split_for_distributed


def transcribe(root: str | Path, kmeans: MiniBatchKMeans, jsonl: str | Path, *, flush_every: int = 1000) -> None:
    root = Path(root)
    jsonl = Path(jsonl)
    files = split_for_distributed(sorted(root.rglob("*.pt")))
    lock = FileLock(f"{jsonl}.lock")
    buffer: list[bytes] = []

    def flush() -> None:
        if not buffer:
            return
        with lock, jsonl.open("ab") as f:
            f.write(b"".join(buffer))
        buffer.clear()

    for path in tqdm(files):
        fileid = str(path.relative_to(root)).removesuffix(".pt")
        units = kmeans.predict(torch.load(path).numpy()).tolist()
        buffer.append(orjson.dumps({"file": fileid, "units": units}, option=orjson.OPT_APPEND_NEWLINE))
        if len(buffer) >= flush_every:
            flush()
    flush()


if __name__ == "__main__":
    import argparse

    import joblib

    parser = argparse.ArgumentParser(description="Inference of discrete units")
    parser.add_argument("root_features", type=Path, help="Root directory containing feature files")
    parser.add_argument("path_kmeans", type=Path, help="Path to the trained K-means model")
    parser.add_argument("output_jsonl", type=Path, help="Path to the output JSONL file with units")
    args = parser.parse_args()
    transcribe(args.root_features, joblib.load(args.path_kmeans), args.output_jsonl)
