## Steering Results
success
False    85
True     15
## Arguments
outdir_idx : 18
dataset : franlucc/starcoderbase-1b-completions_typeinf_analysis
model : /home/arjun/models/starcoderbase-1b
correct_prog_threshold : 100
correct_type_threshold : 100
incorrect_prog_threshold : 100
incorrect_type_threshold : 100
batch_size : 2
patch_mode : add
n_eval : 100
tokens_to_patch : ['<fim_suffix>', '<fim_middle>']
layers_to_patch : [10, 11, 12, 13, 14]
module_to_patch : block_out
additional_filter : False

Eval type distribution
label
any            29
string         21
number          8
unknown         5
void            5
Arrow           4
T               4
object          2
A               2
symbol          2
Error           2
boolean         2
Board           1
Function        1
N               1
Pitch           1
Vector          1
ValueType       1
Object          1
E               1
F               1
Row             1
State           1
ArrayBuffer     1
S               1
Date            1
Name: count, dtype: int64