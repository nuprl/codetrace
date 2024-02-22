## Steering Results
success
False    24
True      1
## Arguments
outdir_idx : 25
dataset : franlucc/stenotype-eval-renamed-v4
model : /home/arjun/models/starcoderbase-1b
correct_prog_threshold : 1000
correct_type_threshold : 1000
incorrect_prog_threshold : 1000
incorrect_type_threshold : 1000
batch_size : 2
patch_mode : add
n_eval : 25
tokens_to_patch : ['<fim_suffix>', '<fim_middle>']
layers_to_patch : [10, 11, 12, 13, 14]
module_to_patch : block_out
additional_filter : False
do_ood_eval : True

Eval type distribution
label
1    16
0     5
2     4
Name: count, dtype: int64