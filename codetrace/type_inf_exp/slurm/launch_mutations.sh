#!/bin/bash
dir="/home/franlucc/projects/codetrace/codetrace/type_inf_exp/scripts"
echo "Py mutations all"
python "${dir}/py_fast_steering_builder.py" --completions-ds franlucc/starcoderbase-7b-py_completions --new-ds-name franlucc/py_all_mutations --model bigcode/starcoderbase-7b --model-name starcoderbase-7b --gpu 3 --mutations rename_vars rename_types remove_type_annotations
rm -r ~/.cache/huggingface/datasets
# python "${dir}/py_fast_steering_builder.py" --completions-ds franlucc/starcoderbase-7b-py_completions --new-ds-name franlucc/py_rename_vars --model bigcode/starcoderbase-7b --model-name starcoderbase-7b --gpu 3 --mutations rename_vars
# python "${dir}/py_fast_steering_builder.py" --completions-ds franlucc/starcoderbase-7b-py_completions --new-ds-name franlucc/py_rename_types --model bigcode/starcoderbase-7b --model-name starcoderbase-7b --gpu 3 --mutations rename_types
# python "${dir}/py_fast_steering_builder.py" --completions-ds franlucc/starcoderbase-7b-py_completions --new-ds-name franlucc/py_delete --model bigcode/starcoderbase-7b --model-name starcoderbase-7b --gpu 3 --mutations remove_type_annotations
# python "${dir}/py_fast_steering_builder.py" --completions-ds franlucc/starcoderbase-7b-py_completions --new-ds-name franlucc/py_all_renames --model bigcode/starcoderbase-7b --model-name starcoderbase-7b --gpu 3 --mutations rename_vars rename_types
# python "${dir}/py_fast_steering_builder.py" --completions-ds franlucc/starcoderbase-7b-py_completions --new-ds-name franlucc/py_vars_and_delete --model bigcode/starcoderbase-7b --model-name starcoderbase-7b --gpu 3 --mutations rename_vars remove_type_annotations
# python "${dir}/py_fast_steering_builder.py" --completions-ds franlucc/starcoderbase-7b-py_completions --new-ds-name franlucc/py_types_and_delete --model bigcode/starcoderbase-7b --model-name starcoderbase-7b --gpu 3 --mutations rename_types remove_type_annotations
