#!/bin/bash
MODEL=$1
OUTPUT_DIR=$2
LANG=$3
MUTATION=$4
LAYERS=$5

export VLLM_LOGGING_LEVEL="ERROR"

if [ "$LANG" = "py" ]; then
    SOURCE_DATASET="nuprl-staging/py_typeinf_fim"
else
    SOURCE_DATASET="nuprl-staging/ts_typeinf_fim"
fi

# 1. do completions
completions_ds="${OUTPUT_DIR}/${LANG}_completions"
if [ ! -d "$completions_ds" ]; then
    python3 -m codetrace.scripts.completions \
            --model $MODEL \
            --prompt-ds $SOURCE_DATASET \
            --new-ds-name $completions_ds
fi

# 2. do mutations

# Example mutations: --mutations="vars,types,delete"
# Choose any combination of [vars, types, delete] where:
# - vars: rename variables
# - types: rename types with aliases
# - delete: remove type annotations

mutations_ds="${OUTPUT_DIR}/${LANG}_${MUTATION}_mutations"
if [ ! -d "$mutations_ds" ]; then
    python3 -m codetrace.scripts.mutate_dataset \
            --model $MODEL \
            --mutated-ds $mutations_ds \
            --completions-ds $completions_ds \
            --lang $LANG \
            --mutations="$MUTATION"
fi

# 3. do steering

# Example layers: --layers="10,11,12"

echo "Layers ${LAYERS}" 
steering_dir="${OUTPUT_DIR}/${LANG}_${MUTATION}_results"
python3 -m codetrace.scripts.launch_steer \
    --model $MODEL \
    --candidates $mutations_ds \
    --output-dir $steering_dir \
    --layers="$LAYERS" \
    --steer-name "steering_split" \
    --test-name "test_split" \
    --tensor-name "steering_tensor.pt" \
    -n -1 \
    --test-size 100
