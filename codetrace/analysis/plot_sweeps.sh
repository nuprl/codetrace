#!/bin/bash

LANG=("py" "ts")
MODELS=("Llama-3.2-3B-Instruct" "CodeLlama-7b-Instruct-hf" "qwen2p5_coder_7b_base" "starcoderbase-1b" "starcoderbase-7b")
RESULTS=$1
OUTDIR=$2
export PYTHONHASHSEED="42"
for language in $"${LANG[@]}"; do
    for model in $"${MODELS[@]}"; do
        echo "Plotting $language $model"
        python -m codetrace.analysis.plot_sweeps mutations \
            --lang $language \
            --model $model \
            --results-dir $RESULTS \
            --outfile "$OUTDIR/fig_$language-$model.pdf"
    done
done