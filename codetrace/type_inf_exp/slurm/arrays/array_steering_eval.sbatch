#!/bin/bash
#SBATCH --partition=177huntington
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --job-name=eval
#SBATCH --output=%A_%a.out
#SBATCH --error=%A_%a.err
#SBATCH --array=7-9

source /work/arjunguha-research-group/franlucc/venvs/v_codetrace/bin/activate

export PYTHONPATH=/work/arjunguha-research-group/franlucc/projects/codetrace:$PYTHONPATH
export HF_DATASETS_CACHE="/work/arjunguha-research-group/franlucc/.cache"

dir="/work/arjunguha-research-group/franlucc/projects/codetrace/codetrace/type_inf_exp"

echo "Running task $SLURM_ARRAY_TASK_ID"
python "${dir}/scripts/launch_steer.py" "${dir}/sbatch_scripts/args/ts_layer_ablation/args${SLURM_ARRAY_TASK_ID}.json"