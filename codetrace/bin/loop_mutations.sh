#!/bin/bash
MODEL=$1
LANG=$2
set -o pipefail

ALL_MUTATIONS=("vars,delete" "types,delete")

# Use an array to track processed mutations
processed=()

while [ "${#processed[@]}" -lt "${#ALL_MUTATIONS[@]}" ]; do
    echo "Processed mutations: ${#processed[@]}"

    for MUTATIONS in "${ALL_MUTATIONS[@]}"; do
        # Skip already processed mutations
        if [[ " ${processed[@]} " =~ " $MUTATIONS " ]]; then
            continue
        fi

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

        if [[ ! -s "$file" ]]; then
            echo "ERROR: Log file $file is empty or unreadable"
            continue
        fi

        grep_output=$(grep -Eo "Collected [0-9]+ candidates" $file)
        if [[ -z "$grep_output" ]]; then
            echo "No candidates collected in $file. Skipping..."
            continue
        fi

        grep -Eo "Collected [0-9]+ candidates" $file | awk '{if ($3 ~ /^[0-9]+$/ && $3 >= 3500) {exit 0} else {exit 1}}'

        if [ $? -eq 0 ]; then
            echo "Finished mutations $MUTATIONS"
            processed+=("$MUTATIONS")  # Mark this mutation as processed
        else
            echo "Problem with mutations $MUTATIONS"
        fi
    done
done
