#!/bin/bash
#SBATCH --partition=177huntington
#SBATCH --job-name=data_collect
#SBATCH --output=%A_%a.out
#SBATCH --error=%A_%a.err
#SBATCH --array=9-9
#SBATCH --time=00:30:00
#SBATCH --mem=48G

source /work/arjunguha-research-group/franlucc/venvs/v_codetrace/bin/activate

export PYTHONPATH=/work/arjunguha-research-group/franlucc/projects/codetrace:$PYTHONPATH
export HF_DATASETS_CACHE="/work/arjunguha-research-group/franlucc/.cache"

dir="/work/arjunguha-research-group/franlucc/projects/codetrace/codetrace/type_inf_exp"

echo "Running task $SLURM_ARRAY_TASK_ID"
python "${dir}/scripts/launch_steer.py" "${dir}/sbatch_scripts/args/py_data_args_dedup/args${SLURM_ARRAY_TASK_ID}.json"
