#!/bin/bash
#SBATCH --job-name=typesteering_steering
#SBATCH --partition=gpuA40x4
#SBATCH --account=bcbj-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=60G
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00

# place a steering_tensor.pt in the OUTPUT_DIR. Then launch_steer will only compute train/test splits
# and run steering using the precomputed tensor. Be mindful of layers, dedup_threshold parameters
# and that candidates ds is consistent with steering-field.

set -x
set -e

MODEL=$1
DATASET=$2
OUTPUT_DIR=$3
LAYERS=$4
STEERING_FIELD=$5

python3 -m codetrace.scripts.launch_steer \
    --model /work/nvme/bcbj/franlucc/models/$MODEL \
    --candidates $DATASET \
    --output-dir $OUTPUT_DIR \
    --layers=$LAYERS \
    --steer-name steering_split \
    --test-name test_split \
    --tensor-name steering_tensor.pt \
    --steering-field $STEERING_FIELD \
    --test-size 500 \
    --dedup-prog-threshold 10 \
    --dedup-type-threshold -1 \
    -b2 5 \
    --run-steering-splits "test" "rand"