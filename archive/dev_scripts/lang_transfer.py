from codetrace.type_inf_exp.scripts.launch_steer import main as main_steer
import os
from argparse import Namespace

edits = [
    "rename_types",
    "rename_vars",
    "all_renames",
    "all_mutations",
    "delete_annotation",
    "types_and_delete",
    "vars_and_delete"
]

############################################
# FILL HERE
############################################
PATH_TO_TS_DATA = "/mnt/ssd/franlucc/projects/codetrace/data/starcoderbase-1b/typescript/may3_seed0"
PATH_TO_PY_DATA = "/mnt/ssd/franlucc/projects/codetrace/data/starcoderbase-1b/python/may3_seed-0-1"

PATH_TO_TS_RESULTS = "/mnt/ssd/franlucc/projects/codetrace/results/starcoderbase-1b/typescript/ts_steering_on_py"
PATH_TO_PY_RESULTS = "/mnt/ssd/franlucc/projects/codetrace/results/starcoderbase-1b/python/py_steering_on_ts"

TENSOR_NAME = "steering_tensor-2000.pt"
MODEL = "/mnt/ssd/arjun/models/starcoderbase-1b"
BATCHSIZE = 4
SEED = 42
############################################

# ts -> py
# for edit in edits:
#     os.makedirs(f"{PATH_TO_TS_RESULTS}/{edit}", exist_ok=True)
#     args ={
#         "steering_tensor": f"{PATH_TO_TS_DATA}/{edit}/{TENSOR_NAME}",
#         "model": MODEL,
#         "expdir": f"{PATH_TO_TS_RESULTS}/{edit}",
#         "evaldir": f"{PATH_TO_PY_DATA}/{edit}",
#         "batch_size": BATCHSIZE,
#         "max_size": -1, 
#         "shuffle": True,
#         "patch_mode": "add",
#         "tokens_to_patch": ["<fim_middle>"],
#         "layers_to_patch": [10,11,12,13,14],
#         "custom_decoder": False,
#         "multiplier": False,
#         "action": "run_steering",
#         "seed": SEED
#     }
#     args = Namespace(**args)
#     main_steer(args)
    
# py -> ts
for edit in edits:
    os.makedirs(f"{PATH_TO_PY_RESULTS}/{edit}", exist_ok=True)
    args ={
        "steering_tensor": f"{PATH_TO_PY_DATA}/{edit}/{TENSOR_NAME}",
        "model": MODEL,
        "expdir": f"{PATH_TO_PY_RESULTS}/{edit}",
        "evaldir": f"{PATH_TO_TS_DATA}/{edit}",
        "batch_size": BATCHSIZE,
        "max_size": -1, 
        "shuffle": True,
        "patch_mode": "add",
        "tokens_to_patch": ["<fim_middle>"],
        "layers_to_patch": [10,11,12,13,14],
        "custom_decoder": False,
        "multiplier": False,
        "action": "run_steering",
        "seed": SEED
    }
    args = Namespace(**args)
    main_steer(args)