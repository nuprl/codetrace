#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gpus-per-node=1
#SBATCH --partition compute

steering_args=$1

source /data/francesca_lucchetti/venvs/codetrace/bin/activate
export PYTHONPATH=/data/francesca_lucchetti/projects/codetrace:$PYTHONPATH

python /data/francesca_lucchetti/projects/codetrace/codetrace/type_inf_exp/scripts/steering.py $steering_args
