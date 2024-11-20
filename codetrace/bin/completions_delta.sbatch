#!/bin/bash
#SBATCH --job-name=typesteering_completions
#SBATCH --partition=gpuA40x4
#SBATCH --account=bcbj-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=160G
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00

set -x
set -e

# You must run this script in the codetrace directory and in an environment
# in which it works.

MODEL=$1
PROMPTS=$2
LANG=$3

BASE=/work/nvme/bcbj

python3 -m codetrace.scripts.completions \
  --model $BASE/models/$MODEL \
  --prompt-ds $PROMPTS \
  --new-ds-name results/completions-$MODEL-$LANG
