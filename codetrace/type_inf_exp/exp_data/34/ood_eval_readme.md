## Steering Results
4 / 30 = 0.13333333333333333
## Arguments
outdir : 34
dataset : franlucc/stenotype-eval-renamed-v5
model : /home/arjun/models/starcoderbase-1b
correct_prog_threshold : 1000
correct_type_threshold : 1000
incorrect_prog_threshold : 1000
incorrect_type_threshold : 1000
batch_size : 2
patch_mode : add
n_eval : 30
tokens_to_patch : ['<fim_suffix>', '<fim_middle>']
layers_to_patch : [10, 11, 12, 13, 14]
test_size : 0.2

Eval type distribution
Counter({'any': 14, 'string': 4, 'Function': 3, 'unknown': 2, 'undefined': 1, 'T': 1, 'TKey': 1, 'Object': 1, 'Error': 1, 'K': 1, 'number': 1})