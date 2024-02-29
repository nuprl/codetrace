#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 <device>"
    exit 1
fi

device=$1

export CUDA_VISIBLE_DEVICES=$device
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
export PYTHONPATH=/home/franlucc/projects/codetrace:$PYTHONPATH
echo "PYTHONPATH=$PYTHONPATH"
export CUDA_LAUNCH_BLOCKING=1
echo "CUDA_LAUNCH_BLOCKING=$CUDA_LAUNCH_BLOCKING"

