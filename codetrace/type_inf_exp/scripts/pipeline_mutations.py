from argparse import Namespace, ArgumentParser
import datasets
import torch
import itertools
from codetrace.type_inf_exp.scripts.typecheck_ds import main as typecheck_main
import os 
from tqdm import tqdm

def combo_to_name(combo):
    if len(combo) == 3:
        return "all_mutations"
    elif len(combo) == 1:
        name =  combo[0].replace("mutation_", "")
        if name == "rename_type":
            return "rename_types"
        else:
            return name
    elif "mutation_rename_type" in combo and "mutation_rename_vars" in combo:
        return "all_renames"
    elif "mutation_rename_type" in combo:
        return "types_and_delete"
    elif "mutation_rename_vars" in combo:
        return "vars_and_delete"
    else:
        raise ValueError("combo not supported")

def pipeline(args):
    # For every mutation combination, run mutation + typecheck
    mutations = ["mutation_rename_type", "mutation_rename_vars", "mutation_delete_annotation"]
    # every mutation combination
    combinations = []
    for i in range(1, len(mutations)+1):
        combinations += list(itertools.combinations(mutations, i))
    
    if args.do_only_combos is not None:
        combinations = [combo for combo in combinations if combo_to_name(combo) in args.do_only_combos]
    combinations = [combo for combo in combinations if not combo_to_name(combo) in args.skip_combos]
    
    print(f"Running mutation pipeline for {len(combinations)} combinations: {combinations}")

    for mut_combo in tqdm(combinations, desc="Mutating and typechecking", total=len(combinations)):
        mutated_ds_name = f"{args.lang}_{combo_to_name(mut_combo)}_{args.unique_id}"
        
        # step 1: run mutation
        mutate_args = args
        mutate_args.mutations = mut_combo
        mutate_args.new_ds_name = f"{args.hf_prefix}/{mutated_ds_name}"
        mutate_args.actions = ["do_mutate", "do_completions"]
        mutate_args.split = "train"
        mutate_args.gpu = str(os.environ["CUDA_VISIBLE_DEVICES"])
        mutate_args.correct_bool = True
        
        print(f"Running mutations for: {mutate_args.new_ds_name}_{args.model_name}")
        
        mutate_main(mutate_args)
        
        print(f"Running typechecking for: {mutate_args.new_ds_name}_{args.model_name}")
        # step 2: run typecheck
        typecheck_args = Namespace(**{"dsname": mutate_args.new_ds_name + "_" + args.model_name, 
                                    "lang":args.lang, 
                                    "new_ds_name": mutate_args.new_ds_name + f"_{args.model_name}_typechecked",
                                    "column_name":"mutated_program",
                                    "local_dataset":False,
                                    "max_size":-1,
                                    "npm_location":os.environ["NPM_PACKAGES"],
                                    "do_log" : False})
        
        typecheck_main(typecheck_args)
        

if __name__=="__main__":
    datasets.disable_caching()
    print("Caching enabled?:", datasets.is_caching_enabled())
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = ArgumentParser()
    parser.add_argument("--completions-ds", type=str, required=True)
    parser.add_argument("--lang", type=str, required=True, choices=["py", "ts"])
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--unique-id", type=str, required=True)
    parser.add_argument("--hf-prefix", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--do-only-combos", type=str, nargs="+", default=None,
                        choices=["all_mutations", 
                                 "rename_types", 
                                 "rename_vars",
                                 "delete_annotations",
                                 "types_and_delete",
                                 "vars_and_delete",
                                 "all_renames"])
    parser.add_argument("--skip-combos", type=str, nargs="+", default=[],
                        choices=["all_mutations", 
                                 "rename_types", 
                                 "rename_vars",
                                 "delete_annotations",
                                 "types_and_delete",
                                 "vars_and_delete",
                                 "all_renames"])
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--max-size", type=int, default=-1)
    args = parser.parse_args()
    
    assert os.environ["CUDA_VISIBLE_DEVICES"] != "", "Set CUDA_VISIBLE_DEVICES to the GPU you want to use"
    assert os.environ["NPM_PACKAGES"] != "", "Set NPM_PACKAGES to the location of your npm packages"
    print("CUDA_VISIBLE_DEVICES:", os.environ["CUDA_VISIBLE_DEVICES"])
    print("NPM_PACKAGES:", os.environ["NPM_PACKAGES"])

    if args.lang == "ts":
        from codetrace.type_inf_exp.scripts.ts_mutate_ds import main as mutate_main
    else:
        from codetrace.type_inf_exp.scripts.py_mutate_ds import main as mutate_main

    if args.model_name is None:
        args.model_name = args.model.split("/")[-1]
        
    pipeline(args)
