## Steering Results
3 / 22 = 0.13636363636363635
## Arguments
outdir : 29
dataset : franlucc/stenotype-type-inference-renamed-eval
model : /home/arjun/models/starcoderbase-1b
correct_prog_threshold : 1
correct_type_threshold : 1
incorrect_prog_threshold : 1
incorrect_type_threshold : 1
batch_size : 2
patch_mode : add
n_eval : 22
tokens_to_patch : ['<fim_suffix>', '<fim_middle>']
layers_to_patch : [10, 11, 12, 13, 14]
test_size : 0

Eval type distribution
Counter({'boolean': 1, 'Schema': 1, 'Actions': 1, 'number': 1, 'string': 1, 'Key': 1, 'any': 1, 'SearchResult': 1, 'Container': 1, 'RegExp': 1, 'HttpResponse': 1, 'Matrix': 1, 'Language': 1, 'Section': 1, 'G': 1, 'void': 1, 'Id': 1, 'U': 1, 'Command': 1, 'Point': 1, 'Options': 1, 'offsets': 1})