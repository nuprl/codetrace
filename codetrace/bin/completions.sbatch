#!/bin/bash
#SBATCH --job-name=typesteering_mutate_dataset
#SBATCH --partition=gpuA40x4
#SBATCH --account=bcbj-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=60G
#SBATCH --gres=gpu:1
#SBATCH --time=5:00:00

set -x
set -e

# You must run this script in the codetrace directory and in an environment
# in which it works.

MODEL=$1
OUTPUT_DIR=$2
LANG=$3

if [ "$LANG" = "py" ]; then
    SOURCE_DATASET="nuprl-staging/py_typeinf_fim"
else
    SOURCE_DATASET="nuprl-staging/ts_typeinf_fim"
fi

python3 -m codetrace.scripts.completions \
    --model /work/nvme/bcbj/models/$MODEL \
    --prompt-ds $SOURCE_DATASET \
    --new-ds-name results/completions-$LANG-$MODEL
