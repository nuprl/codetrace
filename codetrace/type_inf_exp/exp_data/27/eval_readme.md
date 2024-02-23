## Steering Results
19 / 58 = 0.3275862068965517
## Arguments
outdir : 27
dataset : franlucc/stenotype-type-inference-renamed-eval
model : /home/arjun/models/starcoderbase-1b
correct_prog_threshold : 1000
correct_type_threshold : 1000
incorrect_prog_threshold : 1000
incorrect_type_threshold : 1000
batch_size : 2
patch_mode : add
n_eval : 58
tokens_to_patch : ['<fim_suffix>', '<fim_middle>']
layers_to_patch : [10, 11, 12, 13, 14]
test_size : 0

Eval type distribution
Counter({'string': 14, 'number': 12, 'any': 7, 'boolean': 5, 'RegExp': 2, 'void': 2, 'Schema': 1, 'Actions': 1, 'Key': 1, 'SearchResult': 1, 'Container': 1, 'HttpResponse': 1, 'Matrix': 1, 'Language': 1, 'Section': 1, 'G': 1, 'Id': 1, 'U': 1, 'Command': 1, 'Point': 1, 'Options': 1, 'offsets': 1})