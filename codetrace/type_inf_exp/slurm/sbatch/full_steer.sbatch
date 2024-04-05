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
argsdir="${dir}/slurm/args/starcoderbase-1b/more_exp"

# echo "Making data splits..."
# python "${dir}/scripts/launch_steer.py" "${argsdir}/data_collect/args7.json"

# echo "Making steering tensor..."
# python "${dir}/scripts/launch_steer.py" "${argsdir}/tensor_make/args7.json"

echo "Running steering eval..."
python "${dir}/scripts/launch_steer.py" "${argsdir}/args_py_types_and_delete_with_delete_tensor.json"

# echo "Running steering rand eval..."
# python "${dir}/scripts/launch_steer.py" "${argsdir}/rand_eval/args7.json"

# echo "Running layer ablation..."
# python "${dir}/scripts/launch_steer.py" "${argsdir}/layer_ablation_window5.json"