#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --partition=177huntington

source /work/arjunguha-research-group/franlucc/venvs/vllm/bin/activate

export PYTHONPATH=/work/arjunguha-research-group/franlucc/projects/codetrace:$PYTHONPATH
export HF_DATASETS_CACHE=/work/arjunguha-research-group/franlucc/.cache

python /work/arjunguha-research-group/franlucc/projects/codetrace/codetrace/type_inf_exp/scripts/ts_mutate_ds.py \
--completions-ds franlucc/ts_typeinf_natural_1tok_completions \
--model bigcode/starcoderbase-1b \
--new-ds-name franlucc/ts_typeinf_1tok_types_and_delete --mutations mutation_delete_annotation mutation_rename_type