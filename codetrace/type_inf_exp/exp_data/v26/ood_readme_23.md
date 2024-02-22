## Steering Results
success
True     12
False    11
## Arguments
outdir_idx : 26
dataset : franlucc/stenotype-eval-renamed-v4
model : /home/arjun/models/starcoderbase-1b
correct_prog_threshold : 1000
correct_type_threshold : 1000
incorrect_prog_threshold : 1000
incorrect_type_threshold : 1000
batch_size : 2
patch_mode : add
n_eval : 23
tokens_to_patch : ['<fim_suffix>', '<fim_middle>']
layers_to_patch : [10, 11, 12, 13, 14]
additional_filter : False
do_ood_eval : True

Eval type distribution
label
any          8
string       5
number       3
Actions      1
G            1
void         1
boolean      1
Container    1
Point        1
RegExp       1
Name: count, dtype: int64