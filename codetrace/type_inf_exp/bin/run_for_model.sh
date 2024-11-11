MODEL=$1 # name or path
EXP_DIR=$2 # experiment dir of model
DTYPE=$3
NUM_EXAMPLES=$4

# available gpus stats
ALL_GPUS=()
IFS=',' read -r -a ALL_GPUS <<< "$CUDA_VISIBLE_DEVICES"
NUM_GPUS=${#ALL_GPUS[@]}

LANGUAGES=("py") # ts not supported in mutate for now
export VLLM_LOGGING_LEVEL=ERROR

declare -A SOURCE_DATASETS
SOURCE_DATASETS["py"]="franlucc/py_typeinf_fim"
SOURCE_DATASETS["ts"]="franlucc/ts_typeinf_fim"

mut1=("mutation_delete_annotation" "mutation_rename_type" "mutation_rename_vars")
mut2=("mutation_delete_annotation" "mutation_rename_type")
mut3=("mutation_delete_annotation" "mutation_rename_vars")
mut4=("mutation_rename_type" "mutation_rename_vars")
mut5=("mutation_delete_annotation")
mut6=("mutation_rename_vars")
mut7=("mutation_rename_type")

ALL_MUTATIONS=("${mut1[@]}" "${mut2[@]}" "${mut3[@]}" "${mut4[@]}" "${mut5[@]}" "${mut6[@]}" "${mut7[@]}")

## For a given model, run the entirety of type prediction steering including:
# 1. vanilla completions for the source type prediction dataset
# 2. mutations over successful completions
# 3. running steering: creating steering vectors, applying steering vectors and evaluating
#   on steering and eval splits of mutated data

## This in turn has to be done for:
# 1. all languages: py, ts
# 2. all mutations (and combinations of): rename_types, rename_vars, delete_annotations

## To speed this up as much as possible:
# 1. do not recompute data that is already there so we can resume preempted jobs
# 2. aside from blocking jobs (completions), run each atomic job on different GPUS

completions(){
    local completions_ds="$1"
    local language="$2"
    python3 -m codetrace.type_inf_exp.scripts.completions \
        --new-ds-name $completions_ds \
        --model $MODEL \
        --dtype $DTYPE \
        --prompt-ds ${SOURCE_DATASETS[$language]} \

}


mutate(){
    local mutations="$1"
    local plang="$2"
    local completions_ds="$3"
    local new_ds_name="$4"
    local curr_gpu="$5"

    export CUDA_VISIBLE_DEVICES=$curr_gpu
    # NUM_EXAMPLES*2 to account for type checking post filtering
    python3 -m codetrace.type_inf_exp.scripts.${plang}_incremental_mutate \
        --completions-ds $completions_ds \
        --model $MODEL \
        --dtype $DTYPE \
        --new-ds-name $new_ds_name\
        --num-examples $(($NUM_EXAMPLES*2)) \
        --mutations "${mutations[@]}"
    unset CUDA_VISIBLE_DEVICES
}

typecheck(){
    local mutations_ds="$1" 
    local plang="$2"
    local typechecked_ds="$3"
    local column_name="mutated_generated_text"
    python3 -m codetrace.type_inf_exp.scripts.typecheck_ds \
        --ds-name $mutations_ds \
        --new-ds-name $typechecked_ds \
        --column-name $column_name \
        --lang $plang
}

await_free_gpu(){
    gpus=$1
    while true; do
        # Loop through each GPU in the list
        for gpu in "${gpus[@]}"; do
            # Check if the GPU has any processes running using nvidia-smi
            processes=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader)

            # If no processes are running, the GPU is available
            if [ -z "$processes" ]; then
                echo $gpu
                return 0
            fi
        done
        sleep 5  # Poll every 5 seconds
    done
}

# For each language
for plang in $LANGUAGES; do
    ## Part 1: Completions 
    COMPLETIONS_DIR="${EXP_DIR}/${plang}_completions"
    if [-d $COMPLETIONS_DIR]; then
        echo "[1] Skipping ${plang} completions"
    else
        echo "[1] Running ${plang} completions"
        completions $COMPLETIONS_DIR $plang
    fi

    ## Part 2: incrementally mutate across gpus
    for mutants in ${ALL_MUTATIONS[@]}; do
        MUTATION_DIR="${EXP_DIR}/${plang}_${mutants//mutation_/}"

        if [-d $MUTATION_DIR]; then
            echo "[2] Skipping ${MUTATION_DIR}"
        else
            CURR_GPU=$(await_free_gpu $ALL_GPUS)
            echo "[2] Running ${MUTATION_DIR}"

            mutate $mutants $plang $COMPLETIONS_DIR $MUTATION_DIR $CURR_GPU; \
            echo "[2] Typechecking"; \
            typecheck $MUTATION_DIR $plang "${MUTATION_DIR}_typechecked" &
        fi
    done
    export CUDA_VISIBLE_DEVICES=$ALL_GPUS

    ## Part 3: Steering (if previous was successful)
    # TODO

done

