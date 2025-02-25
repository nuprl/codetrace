#!/bin/bash
MODEL=$1
CANDIDATES=$2
OUTPUT_DIR=$3
LAYERS=$4

python3 -m codetrace.scripts.launch_steer \
    --model $MODEL \
    --candidates $CANDIDATES \
    --output-dir $OUTPUT_DIR \
    --layers=$LAYERS \
    --steer-name steering_split \
    --test-name test_split \
    --tensor-name steering_tensor.pt \
    --test-size 100