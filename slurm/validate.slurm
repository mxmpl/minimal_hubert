#!/bin/bash
#SBATCH -J hubert-validate
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=20:00:00

set -euo pipefail

usage() {
    cat <<EOF
Usage: $0 PATH_MANIFEST PATH_CHECKPOINTS OUTPUT_VALIDATION

Run validation for all existing HuBERT checkpoints.

Arguments:
    PATH_MANIFEST        Path to the validation manifest file
    PATH_CHECKPOINTS     Directory containing all intermediate checkpoints
    OUTPUT_VALIDATION    Path to the output JSONL file with validation losses

Example:
    sbatch $0 manifest-val.csv workdir/hubert-it1 validation-hubert-it1.jsonl
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
PATH_CHECKPOINTS=$2
OUTPUT_VALIDATION=$3

srun python -m minimal_hubert.validate \
    "$PATH_MANIFEST" \
    "$PATH_CHECKPOINTS" \
    "$OUTPUT_VALIDATION"
