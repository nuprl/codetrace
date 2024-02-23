## Steering Results
14 / 26 = 0.5384615384615384
## Arguments
outdir : 36
dataset : franlucc/stenotype-type-inference-renamed-eval-v2
model : /home/arjun/models/starcoderbase-1b
correct_prog_threshold : 1000
correct_type_threshold : 1000
incorrect_prog_threshold : 1000
incorrect_type_threshold : 1000
batch_size : 2
patch_mode : add
n_eval : 26
tokens_to_patch : ['<fim_suffix>', '<fim_middle>']
layers_to_patch : [10, 11, 12, 13, 14]
test_size : 0

Eval type distribution
Counter({'any': 7, 'number': 5, 'string': 3, 'boolean': 2, 'SearchResult': 1, 'HttpResponse': 1, 'Schema': 1, 'Section': 1, 'Container': 1, 'Actions': 1, 'Language': 1, 'RegExp': 1, 'Key': 1})