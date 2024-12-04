#!/bin/bash
MODEL=$1
LANG=$2
NPROC=$3

ALL_MUTATIONS=("types" "vars" "delete" "types_vars" "vars_delete" "types_delete" "delete_vars_types")

for MUTATIONS in "${ALL_MUTATIONS[@]}"; do
    echo $MUTATIONS
    python3 -m codetrace.scripts.typecheck_ds \
        --input-ds nuprl-staging/type-steering \
        --subset mutations-$LANG-$MUTATIONS-$MODEL \
        --split train \
        --lang $LANG \
        --output-ds results/mutations-$LANG-$MUTATIONS-$MODEL  \
        --column-name mutated_program \
        --nproc $NPROC >> /tmp/$LANG-$MUTATIONS-$MODEL.out 2>&1 &
done