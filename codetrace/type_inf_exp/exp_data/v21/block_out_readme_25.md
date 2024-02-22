## Steering Results
success
False    20
True      5
## Arguments
outdir_idx : 21
dataset : franlucc/starcoderbase-1b-completions_typeinf_analysis
model : /home/arjun/models/starcoderbase-1b
correct_prog_threshold : 100
correct_type_threshold : 100
incorrect_prog_threshold : 100
incorrect_type_threshold : 100
batch_size : 2
patch_mode : add
n_eval : 25
tokens_to_patch : ['<fim_suffix>', '<fim_middle>']
layers_to_patch : [10, 11, 12, 13, 14]
module_to_patch : block_out
additional_filter : False

Eval type distribution
label
any            8
string         7
Object         1
State          1
S              1
ArrayBuffer    1
T              1
boolean        1
void           1
Error          1
number         1
symbol         1
Name: count, dtype: int64