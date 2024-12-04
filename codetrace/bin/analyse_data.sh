#!/bin/bash

LANG=("py" "ts")
MODELS=("CodeLlama-7b-Instruct-hf" "qwen2p5_coder_7b_base" "starcoderbase-1b" "starcoderbase-7b")
RESULTS=$1
OUTDIR=$2

for language in $"${LANG[@]}"; do
    for model in $"${MODELS[@]}"; do
        echo "Plotting $language $model"
        python -m codetrace.scripts.data_analysis $language $model $RESULTS "$OUTDIR/fig_$language_$model.pdf"
    done
done