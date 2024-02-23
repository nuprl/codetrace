## Steering Results
2 / 14 = 0.14285714285714285
## Arguments
outdir : 28
dataset : franlucc/stenotype-type-inference-renamed-eval
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
Counter({'boolean': 2, 'string': 2, 'Schema': 1, 'Key': 1, 'any': 1, 'SearchResult': 1, 'Container': 1, 'HttpResponse': 1, 'Command': 1, 'void': 1, 'number': 1, 'RegExp': 1})