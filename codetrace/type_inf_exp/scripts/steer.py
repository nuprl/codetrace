from tabulate import tabulate
from codetrace.utils import *
from einops import rearrange
from argparse import ArgumentParser, Namespace
from collections import Counter
import hashlib
import sys
import pickle
from multiprocessing import cpu_count
from copy import deepcopy
from datasets.utils.logging import enable_progress_bar, disable_progress_bar
import hashlib
from transformers import AutoTokenizer
from codetrace.type_inf_exp.steering import *

def make_steering_data_splits(args):
    """
    Generates fit test splits and saves to disk in shared steering_data directory.
    Logs data info to directory.
    """
    print("[STEP 1] Generating fit test splits...")
    ds = datasets.load_dataset(args.source_dataset, split="train")
    print(ds)
    
    if args.shuffle:
        ds = ds.shuffle(seed=42)
        
    # filter out too large prompts for OOM
    ds = ds.filter(lambda x : len(x["fim_program"]) < 8000, desc="Filter OOM")
    if "<FILL>" in ds["fim_program"][0]:
        args.fim_placeholder = True
    else:
        args.fim_placeholder = False
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if "completions" in args.source_dataset or "caa" in args.source_dataset:
        correct, incorrect, incorrect_eval = fit_test_split_completions(ds,tokenizer, args)
    else:
        correct, incorrect, incorrect_eval = fit_test_split(ds,tokenizer, args)
        
    if args.max_size > 0:
        assert args.shuffle, "Please select shuffle when truncating dataset."
        correct = correct.select(range(args.max_size))
        incorrect = incorrect.select(range(args.max_size))
        incorrect_eval = incorrect_eval.select(range(args.max_size))
        
    print("Correct, incorrect, incorrect ood len:", len(correct), len(incorrect), len(incorrect_eval))
    info_incorrect = get_data_info(incorrect, "incorrect")
    info_correct = get_data_info(correct, "correct")
    info_ood = get_data_info(incorrect_eval, "incorrect_ood")
    
    # save splits to disk
    os.makedirs(args.datadir, exist_ok=True)
    correct.save_to_disk(f"{args.datadir}/correct")
    incorrect.save_to_disk(f"{args.datadir}/incorrect")
    incorrect_eval.save_to_disk(f"{args.datadir}/incorrect_ood")

    # save data info
    with open(f"{args.datadir}/data_info.json", "w") as f:
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
    """
    print("[STEP 2] Computing averages for steering tensor...")
    model = LanguageModel(args.model, device_map="cuda")
    
    correct = datasets.load_from_disk(args.datadir + "/correct")
    incorrect = datasets.load_from_disk(args.datadir + "/incorrect")
    if "<FILL>" in correct["fim_program"][0]:
        args.fim_placeholder = True
    else:
        args.fim_placeholder = False
    
    diff_tensor = get_steering_tensor(model, correct, incorrect, args)
    
    # save steering tensor
    torch.save(diff_tensor, f"{args.datadir}/steering_tensor.pt")

def run_steering(args):
    print("[STEP 3] Running steering eval on incorrect and incorrect ood...")
    model = LanguageModel(args.model, device_map="cuda")
    
    os.makedirs(args.expdir, exist_ok=True)
    with open(f"{args.expdir}/args_steering.json", "w") as f:
        json.dump(vars(args), f, indent=4)
    
    # load steering tensor
    diff_tensor = torch.load(f"{args.datadir}/steering_tensor.pt")
    incorrect = datasets.load_from_disk(f"{args.evaldir}/incorrect")
    steer_on_ds(model, diff_tensor, incorrect, False, args)
    
    incorrect_ood = datasets.load_from_disk(f"{args.evaldir}/incorrect_ood")
    steer_on_ds(model, diff_tensor, incorrect_ood, True, args)
    
    
if __name__ == "__main__":
    steering_args = sys.argv[1]
    with open(steering_args, "r") as f:
        args = json.load(f)
    args = Namespace(**args)
    
    if args.action == "make_steering_data_splits":
        make_steering_data_splits(args)
    elif args.action == "make_steering_tensor":
        make_steering_tensor(args)
    elif args.action == "run_steering":
        run_steering(args)
    else:
        raise ValueError("""Invalid action, please choose from:
                         \t- make_steering_data_splits
                         \t- make_steering_tensor
                         \t- run_steering""")