#!/bin/bash
#SBATCH -J hubert-kmeans
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=20:00:00

set -euo pipefail

usage() {
    cat <<EOF
Usage: $0 ROOT_FEATURES PATH_KMEANS N_CLUSTERS SUBSAMPLE

Train K-means on MFCC or HuBERT features.

Arguments:
    ROOT_FEATURES    Root directory containing feature files
    PATH_KMEANS      Output path for the trained K-means model
    N_CLUSTERS       Number of clusters (vocabulary size)
    SUBSAMPLE        Subsampling factor for training (1 = use all files,
                     N = use every Nth file). Higher values speed up
                     training at the cost of potentially worse clustering.

Examples:
    sbatch $0 features/mfcc/ kmeans-mfcc.joblib 100 10
    sbatch $0 features/hubert/ kmeans-hubert.joblib 500 10
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ $# -lt 4 ]]; then
    echo "Error: Missing required arguments." >&2
    usage
    exit 1
fi

ROOT_FEATURES=$1
PATH_KMEANS=$2
N_CLUSTERS=$3
SUBSAMPLE=$4

srun python -m minimal_hubert.kmeans \
    "$ROOT_FEATURES" \
    "$PATH_KMEANS" \
    "$N_CLUSTERS" \
    --subsample "$SUBSAMPLE"
