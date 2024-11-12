from argparse import ArgumentParser, Namespace
from collections import Counter
from transformers import AutoTokenizer
from codetrace.type_inf_exp.steering import (
    fit_test_split, 
    fit_test_split_completions, 
    get_steering_tensor, 
    steer_on_ds,
    get_data_info
)
from pathlib import Path
import os 
from codetrace.utils import load
from codetrace.fast_utils import get_batches_fast, batched_apply
from multiprocessing import cpu_count
from tqdm import tqdm
import datasets
import json
from nnsight import LanguageModel
import torch
import importlib
import sys
# from codetrace.utils import parse_callable
import pandas as pd

def filter_prompts(
    dataset : datasets.Dataset,
    dedup_prog_threshold : int,
    dedup_type_threshold : int
) -> datasets.Dataset:
    """
    Balance prompts s.t. there is a balanced distribution of labels (program ids and/or types).
    """
    # if -1, set to the max value, aka do not dedup
    if dedup_prog_threshold == -1:
        dedup_prog_threshold = len(dataset)
    if dedup_type_threshold == -1:
        dedup_type_threshold = len(dataset)
        
    # get count of labels
    labels = dataset["fim_type"]
    
    hexsha_count = {h:0 for h in dataset["hexsha"]}
    label_count = {label : 0 for label in labels}
    balanced_prompts = []
    for _,ex in tqdm(enumerate(dataset), desc="Deduping dataset",total=len(dataset)):
        if label_count[ex["fim_type"]] >= dedup_type_threshold and hexsha_count[ex["hexsha"]] >= dedup_prog_threshold: 
            # if label and hexsha are already at threshold, break
            break
        elif label_count[ex["fim_type"]] >= dedup_type_threshold or hexsha_count[ex["hexsha"]] >= dedup_prog_threshold:
            # if hexsha is at threshold, continue
            continue
        
        balanced_prompts.append(ex)
        label_count[ex["fim_type"]] += 1
        hexsha_count[ex["hexsha"]] += 1

    df = pd.DataFrame(balanced_prompts)
    ds = datasets.Dataset.from_pandas(df)
    return ds

def filter_oom(batch):
    new_batch = []
    for b in batch:
        if len(b["fim_program"]) < 8000:
            new_batch.append(b)
    return new_batch

def make_steering_data_splits(
    model:str,
    datadir:str,
    source_dataset:str,
    seed:int=None,
    shuffle:bool=True,
    max_size:int=None,
    test_size:float=0.1,
    dedup_prog_threshold:int=-1,
    dedup_type_threshold:int=-1,
    split=None
):
    """
    Generates fit test splits and saves to disk in shared steering_data directory.
    Logs data info to directory.
    """
    print("[STEP 1] Generating fit test splits...")
    if os.path.exists(f"{datadir}/correct") and os.path.exists(f"{datadir}/incorrect"):
        print("Correct and incorrect splits already exist, skipping...")
        return
    
    ds = load(source_dataset, split=split)
    print(ds)
    
    if shuffle:
        ds = ds.shuffle(seed=seed)
    
    # filter out too large prompts for OOM
    batches = get_batches_fast(ds, cpu_count())
    results = batched_apply(batches, cpu_count(), filter_oom)
    
    def yielder():
        for ex in tqdm(results, desc="yielding", total=len(results)):
            yield ex
            
    ds = datasets.Dataset.from_generator(yielder)

    if "<FILL>" in ds["fim_program"][0]:
        fim_placeholder = True
    else:
        fim_placeholder = False
    
    tokenizer = AutoTokenizer.from_pretrained(model)
    if "mutated_program" not in ds.column_names:
        correct, incorrect, incorrect_eval = fit_test_split_completions(ds,tokenizer, test_size)
    else:
        correct, incorrect, incorrect_eval = fit_test_split(ds,tokenizer, test_size)
        
    if dedup_type_threshold > -1 or dedup_prog_threshold > -1:
        assert shuffle, "Please select shuffle when deduping dataset."
        correct = filter_prompts(correct, dedup_prog_threshold, dedup_type_threshold)
        incorrect = filter_prompts(incorrect, dedup_prog_threshold, dedup_type_threshold)
        if incorrect_eval != None:
            incorrect_eval = filter_prompts(incorrect_eval, dedup_prog_threshold, dedup_type_threshold)

    if max_size > 0:
        assert shuffle, "Please select shuffle when truncating dataset."
        correct = correct.select(range(min(max_size, len(correct))))
        incorrect = incorrect.select(range(min(max_size, len(incorrect))))
        if incorrect_eval != None:
            incorrect_eval = incorrect_eval.select(range(min(max_size, len(incorrect_eval))))
    
    
    # [DONE WITH SPLITS, LOG INFO AND SAVE]
    print("Correct, incorrect:", len(correct), len(incorrect))
    if incorrect_eval:
        print("Incorrect ood len:", len(incorrect_eval))
        
    info_incorrect = get_data_info(incorrect, "incorrect")
    info_correct = get_data_info(correct, "correct")
    info_ood = get_data_info(incorrect_eval, "incorrect_ood")
    
    # save splits to disk
    os.makedirs(datadir, exist_ok=True)
    correct.save_to_disk(f"{datadir}/correct")
    incorrect.save_to_disk(f"{datadir}/incorrect")
    if incorrect_eval:
        incorrect_eval.save_to_disk(f"{datadir}/incorrect_ood")

    # save data info
    with open(f"{datadir}/data_info.json", "w") as f:
        info = [info_incorrect, info_correct, info_ood]
        # add types in info_eval that are not in info_incorrect
        if len(info_ood) > 0:
            types = list(set(info_ood["type_counts"].keys()).difference(set(info_incorrect["type_counts"].keys())))
            ood_types = {t : int(info_ood["type_counts"][t]) for t in types}
            info.append({"ood_types" : ood_types})
        json.dump(info, f, indent=4)

def make_steering_tensor(args):
    """
    Load datasets saved to disk.
    Compute correct and incorrect steering tensors, save to shared steering_data directory.
    Create steering tensor from averages
    load if it exists.
    """
    print("[STEP 2] Computing averages for steering tensor...")
    model = LanguageModel(args.model, device_map="cuda")
    
    correct = datasets.load_from_disk(args.datadir + "/correct")
    incorrect = datasets.load_from_disk(args.datadir + "/incorrect")
    print(correct, incorrect)
    
    if args.shuffle:
        correct = correct.shuffle(seed=args.seed)
        incorrect = incorrect.shuffle(seed=args.seed)
        
    if args.max_size > 0:
        assert args.shuffle, "Please select shuffle when truncating dataset."
        correct = correct.select(range(min(args.max_size, len(correct))))
        incorrect = incorrect.select(range(min(args.max_size, len(incorrect))))
        
    if "<FILL>" in correct["fim_program"][0]:
        args.fim_placeholder = True
    else:
        args.fim_placeholder = False
    
    diff_tensor = get_steering_tensor(model, correct, incorrect, args)
    
    # save steering tensor
    torch.save(diff_tensor, f"{args.datadir}/{args.steering_tensor_name}")

def run_steering(args):
    print("[STEP 3] Running steering eval on incorrect and incorrect ood...")
    model = LanguageModel(args.model, device_map="cuda")
    
    os.makedirs(args.expdir, exist_ok=True)
    with open(f"{args.expdir}/args_steering.json", "w") as f:
        json.dump(vars(args), f, indent=4)
    
    # load steering tensor
    diff_tensor = torch.load(args.steering_tensor)
    if args.multiplier != False:
        diff_tensor *= args.multiplier
        
    def _eval(model, diff_tensor, dirname, args):
        eval_ds = datasets.load_from_disk(f"{args.evaldir}/{dirname}")

        if args.shuffle:
            eval_ds = eval_ds.shuffle(seed=args.seed)
        if args.max_size > 0:
            assert args.shuffle, "Please select shuffle when truncating dataset."
            eval_ds = eval_ds.select(range(min(args.max_size, len(eval_ds))))
        
        ood_flag = ""
        if dirname == "incorrect_ood":
            ood_flag = "ood_"
        elif dirname == "correct":
            ood_flag = "correct_"
        steer_on_ds(model, diff_tensor, eval_ds, ood_flag, args)
    
    # incorrect ood eval
    
    if os.path.exists(Path(f"{args.evaldir}/incorrect_ood")):
        _eval(model, diff_tensor, "incorrect_ood", args)
    # incorrect eval
    _eval(model, diff_tensor, "incorrect", args)
    # _eval(model, diff_tensor, "correct", args)

def run_layer_ablation(args):
    """
    The difference between this and run_steering is that this takes the length of layers
    to patch P and performs an ablation from [0...P] all the way to [N-P...N].
    """
    print("[STEP 3 - ABLATION] Running layer ablation steering eval on incorrect and incorrect ood...")
    model = LanguageModel(args.model, device_map="cuda")
    os.makedirs(args.expdir, exist_ok=True)
    
    # load steering tensor
    diff_tensor = torch.load(args.steering_tensor)
    
    # load eval data
    if os.path.exists(Path(f"{args.evaldir}/incorrect_ood")):
        ood_eval_ds = datasets.load_from_disk(f"{args.evaldir}/incorrect_ood")
    fit_eval_ds = datasets.load_from_disk(f"{args.evaldir}/incorrect")
    
    def _eval(model, diff_tensor, do_ood, args):
        if do_ood:
            eval_ds = ood_eval_ds
        else:
            eval_ds = fit_eval_ds
            
        if args.shuffle:
            eval_ds = eval_ds.shuffle(seed=args.seed)
        if args.max_size > 0:
            assert args.shuffle, "Please select shuffle when truncating dataset."
            eval_ds = eval_ds.select(range(min(args.max_size, len(eval_ds))))
            
        steer_on_ds(model, diff_tensor, eval_ds, do_ood, args)
    
    original_dir = args.expdir
    window_size = len(args.layers_to_patch)
    all_layers = list(range(diff_tensor.shape[0]))
    # create sliding window of layers
    zipped = list(zip([all_layers[i:i+window_size] for i in all_layers]))
    windows_to_patch = [j[0] for j in zipped if len(j[0]) == window_size]
    
    for window in windows_to_patch:
        print(f"Running layer ablation {window}")
        args.layers_to_patch = window
        args.expdir = f"{original_dir}/layer_{'-'.join([str(w) for w in window])}"
        os.makedirs(args.expdir, exist_ok=True)
        
        with open(f"{args.expdir}/args_steering.json", "w") as f:
            json.dump(vars(args), f, indent=4)
        
        # only do OOD for now
        # incorrect ood eval
        if os.path.exists(Path(f"{args.evaldir}/incorrect_ood")):
            _eval(model, diff_tensor, True, args)
            
        # # incorrect eval
        # _eval(model, diff_tensor, False, args)
    
                          
def main(args):
    # parse callable from tokens_to_patch if it exists
    if getattr(args, "tokens_to_patch", None) != None:
        # check if args.tokens_to_patch is an importable function
        # args.tokens_to_patch = parse_callable(args.tokens_to_patch)
        raise NotImplementedError("No parse callable")
    
    action = args.__dict__.pop("action")  
    if action == "make_steering_data_splits":
        make_steering_data_splits(**args.__dict__)
    elif action == "make_steering_tensor":
        make_steering_tensor(**args.__dict__)
    elif action == "run_steering":
        run_steering(**args.__dict__)
    elif action == "layer_ablation":
        run_layer_ablation(**args.__dict__)
    else:
        raise ValueError("""Invalid action, please choose from:
                        \t- make_steering_data_splits
                        \t- make_steering_tensor
                        \t- run_steering
                        \t- layer_ablation""")
        
if __name__ == "__main__":
    steering_args = sys.argv[1]
    with open(steering_args, "r") as f:
        args = json.load(f)
    args = Namespace(**args)
    print(f"Passed args:\n{args}")
    main(args)