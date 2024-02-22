## Steering Results
success
True     17
False     8
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

Eval type distribution
label
string     8
any        6
number     2
void       2
Actions    1
boolean    1
Color      1
Options    1
Point      1
Schema     1
RegExp     1
Name: count, dtype: int64