## Steering Results
5 / 14 = 0.35714285714285715
## Arguments
outdir : 35
dataset : franlucc/stenotype-type-inference-renamed-eval-v2
model : /home/arjun/models/starcoderbase-1b
correct_prog_threshold : 1000
correct_type_threshold : 1000
incorrect_prog_threshold : 1000
incorrect_type_threshold : 1000
batch_size : 2
patch_mode : add
n_eval : 14
tokens_to_patch : ['<fim_suffix>', '<fim_middle>']
layers_to_patch : [10, 11, 12, 13, 14]
test_size : 0.2

Eval type distribution
Counter({'number': 2, 'string': 2, 'RegExp': 2, 'boolean': 2, 'Actions': 1, 'void': 1, 'any': 1, 'Key': 1, 'Command': 1, 'SearchResult': 1})
