## Steering Results
11 / 30 = 0.36666666666666664
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
n_eval : 30
tokens_to_patch : ['<fim_suffix>', '<fim_middle>']
layers_to_patch : [10, 11, 12, 13, 14]
test_size : 0

Eval type distribution
Counter({'string': 9, 'number': 5, 'any': 2, 'boolean': 2, 'RegExp': 2, 'Schema': 1, 'Container': 1, 'Options': 1, 'SearchResult': 1, 'Key': 1, 'Point': 1, 'U': 1, 'HttpResponse': 1, 'Section': 1, 'Id': 1})