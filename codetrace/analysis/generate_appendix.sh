#!/bin/bash

export PYTHONHASHSEED="42"
RESULTS_DIR=$1
OUTDIR=$2

ALL_MODELS=("starcoderbase-7b" "Llama-3.2-3B-Instruct" "CodeLlama-7b-Instruct-hf" "qwen2p5_coder_7b_base" "starcoderbase-1b" )
ALL_LANGS=("py" "ts")
ALL_INTERVALS=(1 3 5)

# Splits: all models, langs, intervals

# for model in "${ALL_MODELS[@]}"; do
#     for lang in "${ALL_LANGS[@]}"; do
#         for interval in "${ALL_INTERVALS[@]}"; do
#             python -m codetrace.analysis.plot_fig_splits \
#                 $RESULTS_DIR \
#                 $OUTDIR \
#                 --model $model \
#                 --lang $lang \
#                 --interval $interval
#         done
#     done
# done


# All models: all langs, intervals
for interval in "${ALL_INTERVALS[@]}"; do
    for lang in "${ALL_LANGS[@]}"; do
        python -m codetrace.analysis.plot_fig_all_models \
            $RESULTS_DIR \
            $OUTDIR \
            --lang $lang \
            --interval $interval
    done
done


# Layer abl: all models, langs
for model in "${ALL_MODELS[@]}"; do
    for lang in "${ALL_LANGS[@]}"; do
        python -m codetrace.analysis.plot_fig_layer_ablations \
            $RESULTS_DIR \
            $OUTDIR \
            --model $model \
            --lang $lang
    done
done

# Lang transfer: all langs (py->ts, ts->py), models, intervals
for model in "${ALL_MODELS[@]}"; do
    for lang in "${ALL_LANGS[@]}"; do
        python -m codetrace.analysis.plot_fig_lang_transfer \
            $RESULTS_DIR \
            $OUTDIR \
            --model $model \
            --lang $lang \
            --interval 5 #only did 5
    done
done
