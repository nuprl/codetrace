#!/bin/bash
MODEL=$1
TYPECHECKED_DIR=$2

LANGS=("py" "ts")
ALL_MUTATIONS=("types" "vars" "delete" "types_vars" "vars_delete" "types_delete" "delete_vars_types")
INTERVALS=("0")

for MUTATIONS in "${ALL_MUTATIONS[@]}"; do
    for LANG in "${LANGS[@]}"; do
        for LAYER in "${INTERVALS[@]}"; do
            echo $MODEL $MUTATIONS $LANG $LAYER
            python ~/projects/codetrace/codetrace/tests/check_typechecked_test.py \
                "$TYPECHECKED_DIR/steering-$LANG-$MUTATIONS-$LAYER-$MODEL"
        done
    done
done