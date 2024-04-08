#!/bin/bash
dir="/home/franlucc/projects/codetrace/codetrace/type_inf_exp/scripts"
rm -r ~/.cache/huggingface/datasets
python "${dir}/ts_mutate_ds.py" --completions-ds franlucc/ts_typeinf_7b_completions_v2 --new-ds-name franlucc/ts_types_and_delete_v2 --model bigcode/starcoderbase-7b --model-name starcoderbase-7b --gpu 3 --mutations mutation_rename_type mutation_delete_annotation
rm -r ~/.cache/huggingface/datasets
