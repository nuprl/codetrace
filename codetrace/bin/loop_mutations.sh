#!/bin/bash
MODEL=$1
LANG=$2
set -o pipefail

ALL_MUTATIONS=("delete,vars,types" "vars,delete" "types,delete" "types,vars" "vars" "types" "delete")

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
            --completions-ds results/completions-$LANG-$MODEL \
            --mutated-ds results/mutations-$LANG-$MUTATIONSUNDERSCORED-$MODEL \
            --lang $LANG \
            --batch-size 50 \
            --mutations=$MUTATIONS  >> $file 2>&1

        python3 /home/franlucc/projects/codetrace/codetrace/bin/check_candidates.py "$file"

        if [ $? -eq 0 ]; then
            echo "[PASS] $file has >= 3500 candidates"
            echo "Collected sufficient candidates for mutations $MUTATIONS."
            processed+=("$MUTATIONS")
        else
            echo "[FAIL] $file has < 3500 candidates"
            echo "Not enough candidates for $MUTATIONS."
        fi
    done
done
