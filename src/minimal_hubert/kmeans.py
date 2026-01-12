import logging
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

from .utils import slurm_job_tmpdir

logger = logging.getLogger()


def infer_dimension(root: str | Path) -> int:
    return torch.load(next(Path(root).rglob("*.pt")), map_location="cpu", mmap=True).size(-1)


def build_mmap_features(
    root: str | Path,
    filename: str,
    *,
    dim: int,
    seed: int,
    subsample: int,
) -> np.ndarray:
    files = sorted(Path(root).rglob("*.pt"))
    rng = np.random.default_rng(seed=seed)
    files = rng.choice(files, len(files) // subsample)
    rows = 0
    for path in tqdm(files, "Infer concatenated features shape"):
        shape = torch.load(path, map_location="cpu", mmap=True).shape
        assert len(shape) == 2 and shape[1] == dim
        rows += shape[0]
    logger.info("Shape: %s", (rows, dim))
    mmap = np.memmap(filename, dtype=np.float32, mode="w+", shape=(rows, dim))
    offset = 0
    for path in tqdm(files, "Concatenating features"):
        tensor = torch.load(path)
        length = tensor.size(0)
        mmap[offset : offset + length] = tensor.detach().numpy()
        offset += length
    return mmap


def build_kmeans(n_clusters: int, *, seed: int) -> MiniBatchKMeans:
    return MiniBatchKMeans(
        n_clusters,
        init="k-means++",
        max_iter=100,
        batch_size=10_000,
        verbose=1,
        compute_labels=False,
        random_state=seed,
        tol=0.0,
        max_no_improvement=100,
        n_init=20,
        reassignment_ratio=0.0,
    )


def fit_kmeans(
    root: str | Path,
    n_clusters: int,
    *,
    seed: int,
    subsample: int = 1,
    tmpdir: str | Path | None = None,
) -> MiniBatchKMeans:
    with NamedTemporaryFile(dir=tmpdir, suffix=".dat") as memmap_filename:
        dim = infer_dimension(root)
        features = build_mmap_features(root, memmap_filename.name, dim=dim, seed=seed, subsample=subsample)
        kmeans = build_kmeans(n_clusters, seed=seed).fit(features)
        inertia = -kmeans.score(features) / len(features)
        logger.info("Final inertia: %s", inertia)
    return kmeans


if __name__ == "__main__":
    import argparse

    import joblib

    parser = argparse.ArgumentParser(description="Train K-means on MFCC or HuBERT features")
    parser.add_argument("root_features", type=Path, help="Root directory containing feature files")
    parser.add_argument("path_kmeans", type=Path, help="Output path for the trained K-means model")
    parser.add_argument("n_clusters", type=int, help="Number of clusters")
    parser.add_argument(
        "--subsample",
        type=int,
        default=1,
        help="Subsampling factor: use every Nth file for training (default: 1 = all files).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--tmpdir",
        type=str,
        default=slurm_job_tmpdir(),
        help="Where to store temporary concatenated features",
    )
    args = parser.parse_args()
    kmeans = fit_kmeans(args.root_features, args.n_clusters, seed=args.seed, tmpdir=args.tmpdir)
    joblib.dump(kmeans, args.path_kmeans)
