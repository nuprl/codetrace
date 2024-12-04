#!/bin/bash
MODEL=$1
LANG=$2
set -o pipefail

ALL_MUTATIONS=("types,vars" "vars" "types" "delete" "delete,vars,types" "vars,delete" "types,delete")

counter=0
while [ $counter -lt "${#ALL_MUTATIONS[@]}" ]; do
    echo "Counter $counter"

    for MUTATIONS in "${ALL_MUTATIONS[@]}"; do
        MUTATIONSUNDERSCORED=$(echo $MUTATIONS | sed 's/,/_/g')
        echo $MUTATIONSUNDERSCORED
        file=/tmp/mutations-$LANG-$MUTATIONSUNDERSCORED-$MODEL.out

        VLLM_LOGGING_LEVEL=ERROR python3 -m codetrace.scripts.mutate_dataset \
            --model /mnt/ssd/franlucc/models/$MODEL \
            --tokenizer /mnt/ssd/franlucc/models/$MODEL \
            --completions-ds nuprl-staging/type-steering \
            --subset completions-$LANG-$MODEL \
            --split train \
            --mutated-ds results/mutations-$LANG-$MUTATIONSUNDERSCORED-$MODEL \
            --lang $LANG \
            --batch-size 50 \
            --mutations=$MUTATIONS  >> $file 2>&1

        grep -Eo "Collected [0-9]+ candidates" $file | awk '{if ($3 >= 3500) {exit 0} else {exit 1}}'

        if [ $? -eq 0 ]; then
            counter=$((counter + 1))
            echo "Finished mutations $MUTATIONS, counter: $counter"
        else
            echo "Problem with mutations $MUTATIONS"
        fi
    done
done