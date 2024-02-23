## Steering Results
8 / 100 = 0.08
## Arguments
outdir : 37
dataset : franlucc/stenotype-type-inference-renamed-eval-v3
model : /home/arjun/models/starcoderbase-1b
correct_prog_threshold : 1000
correct_type_threshold : 1000
incorrect_prog_threshold : 1000
incorrect_type_threshold : 1000
batch_size : 2
patch_mode : add
n_eval : 100
tokens_to_patch : ['<fim_suffix>', '<fim_middle>']
layers_to_patch : [10, 11, 12, 13, 14]
test_size : 0

Eval type distribution
Counter({'any': 44, 'string': 13, 'number': 6, 'unknown': 5, 'T': 3, 'Function': 3, 'ValueType': 3, 'void': 3, 'Error': 2, 'undefined': 1, 'TKey': 1, 'Object': 1, 'K': 1, 'object': 1, 'Board': 1, 'L': 1, 'Currency': 1, 'Q': 1, 'Position': 1, 'Pitch': 1, 'S': 1, 'ArrayBuffer': 1, 'Arrow': 1, 'Date': 1, 'A': 1, 'this': 1, 'Symbol': 1})