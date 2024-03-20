#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=10
#SBATCH --job-name=pt_eval
#SBATCH --output=%A_%a.out
#SBATCH --error=%A_%a.err
#SBATCH --partition=177huntington

dir="/work/arjunguha-research-group/franlucc/projects/codetrace/codetrace/type_inf_exp"

source /work/arjunguha-research-group/franlucc/venvs/v_codetrace/bin/activate

export PYTHONPATH=/work/arjunguha-research-group/franlucc/projects/codetrace:$PYTHONPATH
export HF_DATASETS_CACHE=/work/arjunguha-research-group/franlucc/.cache

python "${dir}/scripts/launch_steer.py" "${dir}/sbatch_scripts/args/more_exp/args_ts_with_py_types.json"