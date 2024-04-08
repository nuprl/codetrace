#!/bin/bash
dir="/home/franlucc/projects/codetrace/codetrace/type_inf_exp/scripts"
python "${dir}/ts_mutate_ds.py" --completions-ds franlucc/ts_typeinf_7b_completions_v2 --new-ds-name franlucc/ts_delete --model bigcode/starcoderbase-7b --model-name starcoderbase-7b --gpu 2 --mutations mutation_delete_annotation
rm -r ~/.cache/huggingface/datasets
# python "${dir}/ts_mutate_ds.py" --completions-ds franlucc/ts_typeinf_7b_completions_v2 --new-ds-name franlucc/ts_all_renames --model bigcode/starcoderbase-7b --model-name starcoderbase-7b --gpu 2 --mutations mutation_rename_vars mutation_rename_type
# rm -r ~/.cache/huggingface/datasets
# python "${dir}/ts_mutate_ds.py" --completions-ds franlucc/ts_typeinf_7b_completions_v2 --new-ds-name franlucc/ts_rename_vars --model bigcode/starcoderbase-7b --model-name starcoderbase-7b --gpu 2 --mutations mutation_rename_vars
# python "${dir}/ts_mutate_ds.py" --completions-ds franlucc/ts_typeinf_7b_completions_v2 --new-ds-name franlucc/ts_rename_types --model bigcode/starcoderbase-7b --model-name starcoderbase-7b --gpu 2 --mutations mutation_rename_type
