#!/bin/bash
#SBATCH -J hubert-transcribe
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=05:00:00
#SBATCH --array=0-127

set -euo pipefail

usage() {
    cat <<EOF
Usage: $0 ROOT_FEATURES PATH_KMEANS N_CLUSTERS SUBSAMPLE

Inference of discrete units.

Arguments:
    ROOT_FEATURES    Root directory containing feature files
    PATH_KMEANS      Path to the trained K-means model
    OUTPUT_JSONL     Path to the output JSONL file with units

Examples:
    sbatch $0 features/mfcc/ kmeans-mfcc.joblib units-mfcc.jsonl
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ $# -lt 3 ]]; then
    echo "Error: Missing required arguments." >&2
    usage
    exit 1
fi

ROOT_FEATURES=$1
PATH_KMEANS=$2
OUTPUT_JSONL=$3

srun python -m minimal_hubert.transcribe \
    "$ROOT_FEATURES" \
    "$PATH_KMEANS"  \
    "$OUTPUT_JSONL"
