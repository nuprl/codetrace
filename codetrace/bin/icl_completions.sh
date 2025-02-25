#!/bin/bash

declare -A model_paths
model_paths["CodeLlama-7b-Instruct-hf"]="/mnt/ssd/arjun/models/codellama_7b_instruct"
model_paths["Llama-3.2-3B-Instruct"]="/mnt/ssd/franlucc/models/Llama-3.2-3B-Instruct"
model_paths["starcoderbase-1b"]="/mnt/ssd/arjun/models/starcoderbase-1b"
model_paths["starcoderbase-7b"]="/mnt/ssd/franlucc/models/starcoderbase-7b"
model_paths["qwen2p5_coder_7b_base"]="/mnt/ssd/arjun/models/qwen2p5_coder_7b_base"

for key in "${!model_paths[@]}"; do
    echo "$key -> ${model_paths[$key]}"
done

ALL_LANGS=("py" "ts")
ALL_MUTATIONS=( "delete_vars_types" "vars_delete" "types_delete" "types_vars" "vars" "types" "delete" )
ALL_MODELS=( "starcoderbase-7b" "Llama-3.2-3B-Instruct" "CodeLlama-7b-Instruct-hf" "qwen2p5_coder_7b_base" "starcoderbase-1b" )

export VLLM_LOGGING_LEVEL="ERROR"
for MODEL in "${ALL_MODELS[@]}"; do
    for LANG in "${ALL_LANGS[@]}"; do
        for MUT in "${ALL_MUTATIONS[@]}"; do
            echo $MODEL $LANG $MUT
            python -m codetrace.scripts.icl_completions \
                --model ${model_paths[$MODEL]} \
                --prompt-ds /mnt/ssd/franlucc/scratch/type-steering-results/steering-$LANG-$MUT-0_1_2_3_4-$MODEL \
                --new-ds-name results/icl_$MODEL-$LANG-$MUT
            done
        done
    done