#!/bin/bash
#SBATCH --partition=177huntington
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --job-name=data_t
#SBATCH --output=%A_%a.out
#SBATCH --error=%A_%a.err
#SBATCH --array=3-3

source /work/arjunguha-research-group/franlucc/venvs/v_codetrace/bin/activate

export PYTHONPATH=/work/arjunguha-research-group/franlucc/projects/codetrace:$PYTHONPATH
export HF_DATASETS_CACHE="/work/arjunguha-research-group/franlucc/.cache"

dir="/work/arjunguha-research-group/franlucc/projects/codetrace/codetrace/type_inf_exp"

if [ "$SLURM_ARRAY_TASK_ID" -ge 1 ] && [ "$SLURM_ARRAY_TASK_ID" -le 8 ]; then
    echo "Running task $SLURM_ARRAY_TASK_ID"
    python "${dir}/scripts/steer.py" "${dir}/sbatch_scripts/steering_tensor_args/args${SLURM_ARRAY_TASK_ID}.json"
fi