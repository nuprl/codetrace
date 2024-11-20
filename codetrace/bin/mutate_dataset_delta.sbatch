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
MUTATIONS=$2
LANG=$3

# Replace the commas with underscores in the mutations string.
MUTATIONSUNDERSCORED=$(echo $MUTATIONS | sed 's/,/_/g')

python3 -m codetrace.scripts.mutate_dataset \
    --model /work/nvme/bcbj/models/$MODEL \
    --tokenizer /work/nvme/bcbj/models/$MODEL \
    --completions-ds results/completions-$LANG-$MODEL \
    --mutated-ds results/mutations-$LANG-$MUTATIONSUNDERSCORED-$MODEL \
    --lang $LANG \
    --mutations=$MUTATIONS
