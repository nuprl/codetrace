MODEL=$1 # name or path
EXP_DIR=$2 # experiment dir of model

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

NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
LANGUAGES=("py" "ts")

# For each language
for plang in $LANGUAGES; do
    ## Part 1: Completions 
    COMPLETIONS_DIR="${EXP_DIR}/${plang}_completions"
    if [-d $COMPLETIONS_DIR]; do
        echo "[1] Skipping ${plang} completions"
    else
        echo "[1] Running ${plang} completions"
    fi

    ## Part 2: incrementally mutate across gpus
    
done

