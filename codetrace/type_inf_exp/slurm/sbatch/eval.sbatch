#!/bin/bash
#SBATCH --partition=177huntington
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=10
#SBATCH --job-name=eval_scripts
#SBATCH --output=%A_%a.out
#SBATCH --error=%A_%a.err

source /work/arjunguha-research-group/franlucc/venvs/v_codetrace/bin/activate

export PYTHONPATH=/work/arjunguha-research-group/franlucc/projects/codetrace:$PYTHONPATH
export HF_DATASETS_CACHE="/work/arjunguha-research-group/franlucc/.cache"
export TOKENIZERS_PARALLELISM="false"

dir="/work/arjunguha-research-group/franlucc/projects/codetrace/codetrace/type_inf_exp"
argsdir="${dir}/slurm/args/starcoderbase-7b"

echo "Running steering eval..."
python "${dir}/scripts/launch_steer.py" "${argsdir}/python/array/eval/args6.json"
python "${dir}/scripts/launch_steer.py" "${argsdir}/typescript/array/eval/args1.json"
python "${dir}/scripts/launch_steer.py" "${argsdir}/typescript/array/eval/args2.json"
python "${dir}/scripts/launch_steer.py" "${argsdir}/typescript/array/eval/args3.json"
python "${dir}/scripts/launch_steer.py" "${argsdir}/typescript/array/eval/args4.json"
python "${dir}/scripts/launch_steer.py" "${argsdir}/typescript/array/eval/args5.json"
python "${dir}/scripts/launch_steer.py" "${argsdir}/typescript/array/eval/args6.json"

echo "Running steering rand eval..."
python "${dir}/scripts/launch_steer.py" "${argsdir}/typescript/array/rand_eval/args1.json"
python "${dir}/scripts/launch_steer.py" "${argsdir}/typescript/array/rand_eval/args2.json"
python "${dir}/scripts/launch_steer.py" "${argsdir}/typescript/array/rand_eval/args3.json"
python "${dir}/scripts/launch_steer.py" "${argsdir}/typescript/array/rand_eval/args4.json"
python "${dir}/scripts/launch_steer.py" "${argsdir}/typescript/array/rand_eval/args5.json"
python "${dir}/scripts/launch_steer.py" "${argsdir}/typescript/array/rand_eval/args6.json"