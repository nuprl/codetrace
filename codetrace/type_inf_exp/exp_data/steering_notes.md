## Optimal steering config

Run 25
## Steering Results
success
False    21
True     10
32%, 10/31
## Arguments
outdir_idx : 24
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
A              1
T              1
unknown        1
undefined      1
symbol         1
string         1
primitive      1
object         1
number         1
boolean        1
any            1
Vertex         1
Vector         1
ValueType      1
TKey           1
State          1
ArrayBuffer    1
Row            1
Resolved       1
Position       1
Pitch          1
Options        1
Metadata       1
JsonKey        1
Id             1
Function       1
Error          1
Date           1
Currency       1
Board          1
void           1
Name: count, dtype: int64