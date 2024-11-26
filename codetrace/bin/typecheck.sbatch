#!/bin/bash
#SBATCH --job-name=typechecking
#SBATCH --account=bcbj-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --mem=60G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

set -x
set -e

# You must run this script in the codetrace directory and in an environment
# in which it works.

LANG=$1
INPUT_DS=$2
OUTPUT_DS=$3

/work/nvme/bcbj/aguha/venv-x86/bin/python3.11 -m codetrace.scripts.typecheck_ds \
    --input-ds $INPUT_DS \
    --lang $LANG \
    --output-ds $OUTPUT_DS \
    --column-name mutated_program
