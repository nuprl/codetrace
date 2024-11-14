#!/bin/bash
set -e # Halt on errors

# Check for three arguments: model name, dataset, and output path
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <model-name> <dataset> <output-path>"
    exit 1
fi

MODEL_NAME=$1
DATASET=$2
OUTPUT_PATH=$3

# The 25 token limit is what we have used before:
# https://github.com/arjunguha/santacoder_fim_benchmark/blob/master/generation.py#L35
python3 -m prl_ml.batched_lm_generation.vllm_base \
    --model-name $MODEL_NAME \
    --dataset $DATASET \
    --output-dir $OUTPUT_PATH \
    --temperature 0 \
    --batch-size 50 \
    --max-tokens 25 \
    --completion-limit 1 \
    --prompt-keys "prompt" \
    --extra-columns name,canonical_solution,language \
    --stop '[ ]'