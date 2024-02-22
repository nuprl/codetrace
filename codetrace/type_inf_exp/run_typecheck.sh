#!/bin/bash

TARGET_DIR=$1
COUNT=0
NUM_FILES=0

# count number of files in TARGET_DIR
for file in $TARGET_DIR/*.ts; do
    NUM_FILES=$((NUM_FILES+1))
done
echo "Number of files: $NUM_FILES"
mkdir tsc_log
# for every file in TARGET_DIR ending with .ts
# collect a list of files with type errors and output the type errors
file_list=""
for file in $TARGET_DIR/*.ts; do
    # collect only stderr
    npx --cache ~/.npm_packages ts-node --typeCheck $file 2> tsc_log/typecheck_output.txt
    # if there are type errors, add the file to the list
    if [ -s ../log/typecheck_output.txt ]; then
        # save output to ../log/$filename
        # replace separator with underscore
        fname=$(echo $file | sed 's/\//_/g')
        cp tsc_log/typecheck_output.txt tsc_log/$fname
        file_list="$file_list $file"
        # increment COUNT
        COUNT=$((COUNT+1))
    fi
    echo $COUNT
    echo $file
done
echo "Number of files: $NUM_FILES"
echo "Number of files with type errors: $COUNT"
