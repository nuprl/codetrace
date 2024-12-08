#!/bin/bash
MODEL=$1
LANG_TENSOR=$2
LANG_TEST=$3
NLAYERS=$4
INTERVAL=$5

# we assume steering-$LANG-$MUTATIONSUNDERSCORED-0-$MODEL has a tensor
# with all layers and precomputed test split; if not this will error
ALL_MUTATIONS=("vars" "types" "delete" "delete,vars,types" "vars,delete" "types,delete" "types,vars")

for MUTATIONS in $"${ALL_MUTATIONS[@]}"; do
    MUTATIONSUNDERSCORED=$(echo $MUTATIONS | sed 's/,/_/g')
    echo "sbatch codetrace/bin/custom_steering_sweep.py \
--model $MODEL \
--test-ds results/steering-$LANG_TEST-$MUTATIONSUNDERSCORED-0-$MODEL/test_split \
--test-ds-mutations $MUTATIONS \
--steering-field mutated_program \
--steering-tensor results/steering-$LANG_TENSOR-$MUTATIONSUNDERSCORED-0-$MODEL/steering_tensor.pt \
--results-label lang_transfer_${LANG_TENSOR}_ \
--lang $LANG_TEST \
--num-layers $NLAYERS \
--interval $INTERVAL"
    echo
done