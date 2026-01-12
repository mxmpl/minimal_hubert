#!/bin/bash
#SBATCH -J hubert-features
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=05:00:00
#SBATCH --array=0-127

set -euo pipefail

usage() {
    cat <<EOF
Usage: $0 PATH_MANIFEST ROOT_FEATURES TYPE [OPTIONS]

Extract features using MFCC or HuBERT.

Arguments:
    PATH_MANIFEST    Path to the manifest file
    ROOT_FEATURES    Root directory for output features
    TYPE             Feature type: 'mfcc' or 'hubert'

Options for 'hubert':
    PATH_CHECKPOINT  Path to HuBERT checkpoint file
    LAYER            Layer number to extract features from

Examples:
    sbatch $0 manifest.csv features/ mfcc
    sbatch $0 manifest.csv features/ hubert checkpoints/hubert.pt 9
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

PATH_MANIFEST=$1
ROOT_FEATURES=$2

if [[ "$3" == "mfcc" ]]; then
    srun python -m minimal_hubert.features \
    "$PATH_MANIFEST" \
    "$ROOT_FEATURES" \
    --type mfcc

elif [[ "$3" == "hubert" ]]; then
    PATH_CHECKPOINT=$4
    LAYER=$5
    srun python -m minimal_hubert.features \
    "$PATH_MANIFEST" \
    "$ROOT_FEATURES" \
    --type hubert \
    --checkpoint "$PATH_CHECKPOINT" \
    --layer "$LAYER"

else
    echo "Error: Invalid argument '$3'. Expected 'mfcc' or 'hubert'." >&2
    exit 1
fi
