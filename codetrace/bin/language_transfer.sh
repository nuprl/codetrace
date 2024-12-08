#!/bin/bash
MODEL=$1
LANG_TENSOR=$2
LANG_TEST=$3
NLAYERS=$4
INTERVAL=$5

# we assume steering-$LANG-$MUTATIONSUNDERSCORED-0-$MODEL has a tensor
# with all layers and precomputed test split; if not this will error
ALL_MUTATIONS=("vars" "types" "delete" "delete,vars,types" "vars,delete" "types,delete" "types,vars")

for MUTATIONS in $"${ALL_MUTATIONS[@]}"; do
    MUTATIONSUNDERSCORED=$(echo $MUTATIONS | sed 's/,/_/g')
    echo "sbatch codetrace/bin/custom_steering_sweep.py \
--model $MODEL \
--test-ds results/steering-$LANG_TEST-$MUTATIONSUNDERSCORED-0-$MODEL/test_split \
--test-ds-mutations $MUTATIONS \
--steering-field mutated_program \
--steering-tensor results/steering-$LANG_TENSOR-$MUTATIONSUNDERSCORED-0-$MODEL/steering_tensor.pt \
--results-label lang_transfer_${LANG_TENSOR}_ \
--lang $LANG_TEST \
--num-layers $NLAYERS \
--interval $INTERVAL"
    echo
done

# import argparse
# import subprocess

# ALL_MUTATIONS=["vars","types","delete","delete,vars,types","vars,delete","types,delete","types,vars"]

# def model_n_layer(model: str) -> int:
#     if "qwen" in model.lower():
#         return 28
#     elif "codellama" in model.lower():
#         return 32
#     elif "starcoderbase-1b" in model.lower():
#         return 24
#     elif "starcoderbase-7b" in model.lower():
#         return 42
#     elif "llama" in model.lower():
#         return 28
#     else:
#         raise ValueError(f"Model {model} model_n_layer not implemented!")

# def main_with_args(model: str, lang_tensor: str, lang_test: str, interval:int, dry_run:bool):
#     # we assume steering-$LANG-$MUTATIONSUNDERSCORED-0-$MODEL has a tensor
#     # with all layers and precomputed test split; if not this will error

#     nlayers = model_n_layer(model)
    
#     for mutations in ALL_MUTATIONS:
#         mutations_underscored = mutations.replace(",","_")
#         cmd = ["sbatch","codetrace/bin/custom_steering_sweep.py",
#             "--model",model,
#             "--test-ds",f"results/steering-{lang_test}-{mutations_underscored}-0-{model}/test_split",
#             "--test-ds-mutations",mutations,
#             "--steering-field","mutated_generated_text",
#             "--steering-tensor",f"results/steering-{lang_tensor}-{mutations_underscored}-0-{model}/steering_tensor.pt",
#             "--results-label",f"lang_transfer_{lang_tensor}_",
#             "--lang",lang_test,
#             "--num-layers",str(nlayers),
#             "--interval",str(interval)
#         ]

#         if dry_run:
#             print(" ".join(cmd))
#         else:
#             subprocess.run(cmd)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model", type=str, required=True)
#     parser.add_argument("--lang-tensor", type=str, required=True, choices=["py","ts"])
#     parser.add_argument("--lang-test", type=str, required=True, choices=["py","ts"])
#     parser.add_argument("--interval", type=int, required=True, choices=[1,3,5])
#     parser.add_argument("--dry-run", action="store_true")
#     args = parser.parse_args()
#     assert args.lang_tensor != args.lang_test
#     main_with_args(**vars(args))
