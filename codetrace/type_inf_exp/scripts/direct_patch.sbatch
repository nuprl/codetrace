#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gpus-per-node=1
#SBATCH --partition compute

direct_patch_args=$1

source /data/francesca_lucchetti/venvs/codetrace/bin/activate
export PYTHONPATH=/data/francesca_lucchetti/projects/codetrace:$PYTHONPATH

python /data/francesca_lucchetti/projects/codetrace/codetrace/type_inf_exp/scripts/direct_patch.py $direct_patch_args
