#!/bin/bash
MODEL=$1
TYPECHECKED_DIR=$2

LANGS=("py" "ts")
ALL_MUTATIONS=("types" "vars" "delete" "types_vars" "vars_delete" "types_delete" "delete_vars_types")

for MUTATIONS in "${ALL_MUTATIONS[@]}"; do
    for LANG in "${LANGS[@]}"; do
        echo $MODEL $MUTATIONS $LANG
        python ~/projects/codetrace/codetrace/tests/check_typechecked_data.py \
            "$TYPECHECKED_DIR/mutations-$LANG-$MUTATIONS-$MODEL"
    done
done