## Steering Results
success
True     37
False    25
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
n_eval : 62
tokens_to_patch : ['<fim_suffix>', '<fim_middle>']
layers_to_patch : [10, 11, 12, 13, 14]
module_to_patch : block_out
additional_filter : False

Eval type distribution
label
any             14
string          13
number          11
void             3
boolean          3
RegExp           2
Id               1
Key              1
Language         1
Options          1
Point            1
Color            1
Schema           1
SearchResult     1
Section          1
U                1
HttpResponse     1
G                1
Container        1
offsets          1
Command          1
Actions          1
Name: count, dtype: int64