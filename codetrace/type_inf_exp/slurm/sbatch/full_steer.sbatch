#!/bin/bash
#SBATCH --partition=177huntington
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=10
#SBATCH --job-name=full
#SBATCH --output=%A_%a.out
#SBATCH --error=%A_%a.err

source /work/arjunguha-research-group/franlucc/venvs/v_codetrace/bin/activate

export PYTHONPATH=/work/arjunguha-research-group/franlucc/projects/codetrace:$PYTHONPATH
export HF_DATASETS_CACHE="/work/arjunguha-research-group/franlucc/.cache"
export TOKENIZERS_PARALLELISM="false"

dir="/work/arjunguha-research-group/franlucc/projects/codetrace/codetrace/type_inf_exp"
argsdir="${dir}/slurm/args/starcoderbase-1b/proper_caa/py_rename_types"

echo "Making data splits..."
python "${dir}/scripts/launch_steer.py" "${argsdir}/data.json"

echo "Making steering tensor..."
python "${dir}/scripts/launch_steer.py" "${argsdir}/tensors.json"

echo "Running steering eval..."
python "${dir}/scripts/launch_steer.py" "${argsdir}/eval.json"

# echo "Running layer ablation..."
# python "${dir}/scripts/launch_steer.py" "${argsdir}/layer_ablation_window5.json"