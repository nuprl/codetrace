#!/bin/bash
#SBATCH --partition=177huntington
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=10
#SBATCH --job-name=collect
#SBATCH --output=%A_%a.out
#SBATCH --error=%A_%a.err

source /work/arjunguha-research-group/franlucc/venvs/v_codetrace/bin/activate

export PYTHONPATH=/work/arjunguha-research-group/franlucc/projects/codetrace:$PYTHONPATH
export HF_DATASETS_CACHE="/work/arjunguha-research-group/franlucc/.cache"
export TOKENIZERS_PARALLELISM="false"

dir="/work/arjunguha-research-group/franlucc/projects/codetrace/codetrace/type_inf_exp"
modeldir="/work/arjunguha-research-group/arjun/models"
resultsdir="${dir}/results/starcoderbase-1b/python/exp_results_py"
datadir="${dir}/data/starcoderbase-1b/python/exp_data_py"

echo "Running collect activations..."
python "${dir}/scripts/collect_activations.py" --model-name "${modeldir}/starcoderbase-1b" \
--local-result-datasets "${resultsdir}/py_all_mutations_v0_fit/steering_results_ds" "${resultsdir}/py_all_mutations_v0_fit/ood_steering_results_ds" \
--outdir /scratch/lucchetti.f/codetrace_activations/test_py_all_mut \
--batchsize 2 \
--steering-tensor "${datadir}/py_typeinf_all_mutations_v0/steering_tensor-2000.pt" \
--layers-to-patch 10 11 12 13 14 \
--max-size 1000 \
--start-at-index 80