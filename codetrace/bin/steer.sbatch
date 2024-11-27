#!/bin/bash
#SBATCH --job-name=typesteering_steering
#SBATCH --partition=gpuA40x4
#SBATCH --account=bcbj-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=60G
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00

set -x
set -e

MODEL=$1
MUTATIONS=$2
LANG=$3
LAYERS=$4

MUTATIONSUNDERSCORED=$(echo $MUTATIONS | sed 's/,/_/g')
LAYERSUNDERSCORED=$(echo $LAYERS | sed 's/,/_/g')

python3 -m codetrace.scripts.launch_steer \
    --model /work/nvme/bcbj/franlucc/models/$MODEL \
    --candidates "nuprl-staging/type-steering" \
    --subset "mutations-$LANG-$MUTATIONSUNDERSCORED-$MODEL" \
    --split "train" \
    --output-dir "results/steering-$LANG-$MUTATIONSUNDERSCORED-$LAYERSUNDERSCORED-$MODEL" \
    --layers=$LAYERS \
    --steer-name steering_split \
    --test-name test_split \
    --tensor-name steering_tensor.pt \
    -n 3000 \
    --test-size 100