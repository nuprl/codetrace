#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gpus-per-node=1
#SBATCH --partition=177huntington

args=$1

source /work/arjunguha-research-group/franlucc/venvs/v_codetrace/bin/activate
export PYTHONPATH=/work/arjunguha-research-group/franlucc/projects/codetrace:$PYTHONPATH

python /work/arjunguha-research-group/franlucc/projects/codetrace/codetrace/type_inf_exp/scripts/layer_ablation.py $args
