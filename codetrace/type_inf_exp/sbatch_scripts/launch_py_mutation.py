from codetrace.py_fast_steering_builder import main
from argparse import Namespace

# 0. types

# args = {
#     "completions_ds": "franlucc/py_caa",
#     "model": "/home/arjun/models/starcoderbase-1b",
#     "new_ds_name": "franlucc/py_typeinf_rename_types",
#     "mutations": ["rename_types"],
#     "only_completions": False,
#     "split": "train",
#     "max_size": -1
# }
# args = Namespace(**args)
# main(args)

# # 1. delete

# args = {
#     "completions_ds": "franlucc/py_caa",
#     "model": "/home/arjun/models/starcoderbase-1b",
#     "new_ds_name": "franlucc/py_typeinf_delete",
#     "mutations": ["remove_type_annotations"],
#     "only_completions": False,
#     "split": "train",
#     "max_size": -1
# }
# args = Namespace(**args)
# main(args)

# # 2. all_renames

# args = {
#     "completions_ds": "franlucc/py_caa",
#     "model": "/home/arjun/models/starcoderbase-1b",
#     "new_ds_name": "franlucc/py_typeinf_all_renames",
#     "mutations": ["rename_types", "rename_vars"],
#     "only_completions": False,
#     "split": "train",
#     "max_size": -1
# }
# args = Namespace(**args)
# main(args)

# # 3. vars_and_delete

# args = {
#     "completions_ds": "franlucc/py_caa",
#     "model": "/home/arjun/models/starcoderbase-1b",
#     "new_ds_name": "franlucc/py_typeinf_vars_and_delete",
#     "mutations": ["remove_type_annotations", "rename_vars"],
#     "only_completions": False,
#     "split": "train",
#     "max_size": -1
# }
# args = Namespace(**args)
# main(args)

# 4. types_and_delete

print("Running types_and_delete")

args = {
    "completions_ds": "franlucc/py_caa",
    "model": "/home/arjun/models/starcoderbase-1b",
    "new_ds_name": "franlucc/py_typeinf_types_and_delete",
    "mutations": ["remove_type_annotations", "rename_types"],
    "only_completions": False,
    "split": "train",
    "max_size": -1
}
args = Namespace(**args)
main(args)

# 5. all_muts

print("Running all_muts")
args = {
    "completions_ds": "franlucc/py_caa",
    "model": "/home/arjun/models/starcoderbase-1b",
    "new_ds_name": "franlucc/py_typeinf_all_mutations",
    "mutations": ["remove_type_annotations", "rename_types", "rename_vars"],
    "only_completions": False,
    "split": "train",
    "max_size": -1
}

args = Namespace(**args)
main(args)
