#!/bin/bash
#SBATCH -J hubert-abx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=02:00:00
#SBATCH --array=1-12

set -euo pipefail

usage() {
    cat <<EOF
Usage: $0 PATH_ITEM ROOT_AUDIO OUTPUT_ABX PATH_CHECKPOINT

Compute ABX discriminability of HuBERT representations.

Arguments:
    PATH_MANIFEST    Path to the item file
    ROOT_AUDIO       Path to the audio files
    OUTPUT_ABX       Path to the output JSONL file with ABX scores
    PATH_CHECKPOINT  Path to HuBERT checkpoint file

Example:
    sbatch $0 triphone-dev-clean.item ./LibriSpeech/dev-clean hubert-abx.jsonl ./workdir/hubert/best.pt
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

PATH_ITEM=$1
ROOT_AUDIO=$2
OUTPUT_ABX=$3
PATH_CHECKPOINT=$4

srun python minimal_hubert.abx \
  "$PATH_ITEM" \
  "$ROOT_AUDIO" \
  "$OUTPUT_ABX" \
  --checkpoint "$PATH_CHECKPOINT" \
  --layers "$SLURM_ARRAY_TASK_ID" \
  --extension ".wav"
