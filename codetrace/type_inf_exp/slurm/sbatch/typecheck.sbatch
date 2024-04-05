#!/bin/bash
#SBATCH --partition=express
#SBATCH --job-name=typecheck
#SBATCH --output=%A_%a.out
#SBATCH --error=%A_%a.err
#SBATCH --time=1:00:00
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=100

source /work/arjunguha-research-group/franlucc/venvs/v_codetrace/bin/activate
module load nodejs
export PYTHONPATH=/work/arjunguha-research-group/franlucc/projects/codetrace:$PYTHONPATH
export HF_DATASETS_CACHE="/work/arjunguha-research-group/franlucc/.cache"
export TOKENIZERS_PARALLELISM="false"

dir="/work/arjunguha-research-group/franlucc/projects/codetrace/codetrace/type_inf_exp/scripts"
datadir="/work/arjunguha-research-group/franlucc/projects_data/codetrace/results/starcoderbase-1b"
python "${dir}/get_parse_count.py" --list-o-dirs "${datadir}/python/*" --lang py --outfile "${dir}/py_typecheck_fast.csv"
