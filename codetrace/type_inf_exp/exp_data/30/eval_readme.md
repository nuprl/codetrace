## Steering Results
37 / 62 = 0.5967741935483871
## Arguments
outdir : 30
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
test_size : 0

Eval type distribution
Counter({'any': 14, 'string': 13, 'number': 11, 'boolean': 3, 'void': 3, 'RegExp': 2, 'Actions': 1, 'Color': 1, 'Command': 1, 'Container': 1, 'G': 1, 'HttpResponse': 1, 'Id': 1, 'Key': 1, 'Language': 1, 'Options': 1, 'Point': 1, 'Schema': 1, 'SearchResult': 1, 'Section': 1, 'U': 1, 'offsets': 1})