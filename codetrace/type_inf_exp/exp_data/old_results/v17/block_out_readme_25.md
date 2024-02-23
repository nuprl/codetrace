## Steering Results
success
False    17
True      8
## Arguments
outdir_idx : 17
dataset : franlucc/starcoderbase-1b-completions_typeinf_analysis
model : /home/arjun/models/starcoderbase-1b
correct_prog_threshold : 1
correct_type_threshold : 1
incorrect_prog_threshold : 1
incorrect_type_threshold : 1
batch_size : 2
patch_mode : add
n_eval : 25
tokens_to_patch : ['<fim_suffix>', '<fim_middle>']
layers_to_patch : [10, 11, 12, 13, 14]
module_to_patch : block_out
additional_filter : False

Eval type distribution
label
T              1
Function       1
primitive      1
symbol         1
Id             1
Board          1
TKey           1
number         1
unknown        1
Error          1
any            1
Currency       1
Date           1
A              1
Options        1
Row            1
Vector         1
undefined      1
Vertex         1
ArrayBuffer    1
Metadata       1
Position       1
object         1
string         1
ValueType      1
Name: count, dtype: int64