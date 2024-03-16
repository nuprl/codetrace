from tabulate import tabulate
from codetrace.type_inf_exp.request_patch import *
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

def keep_columns(ds, cols):
    columns = [c for c in ds.column_names if c not in cols]
    return ds.remove_columns(columns)

def dedup_ds_by_key(ds, key):
    """
    Dedup ds by key. Picks the first occurence of key.
    """
    seen = set()
    new_ds = []
    for x in ds:
        if not x[key] in seen:
            new_ds.append(x)
            seen.add(x[key])
    return datasets.Dataset.from_pandas(pd.DataFrame(new_ds))


def get_field_subset(incorrect, field, test_len, reverse):
    """
    Return a subset of incorrect prompts that have the least/most count of field.
    stop when test_len is reached.
    """
     # set aside some incorrect prompts
    field_counts = Counter(incorrect[field])
    # accumulate from the least/most count until threshold test_size is achieved
    sorted_field = sorted(list(incorrect), key= lambda x : field_counts[x[field]], reverse=reverse)
    f_subset = []
    for x in sorted_field:
        f_subset.append(x)
        if len(f_subset) > test_len:
            break
    return f_subset


def make_source_program_ood(correct, incorrect, args):
    """
    OOD should have different set of source programs
    """
    if args.test_size > 1:
        test_len = args.test_size
    else:
        test_len = int(len(incorrect) * args.test_size)
        
    ood_incorrect = get_field_subset(incorrect, "hexsha", test_len, reverse=False)
    ood_incorrect = datasets.Dataset.from_pandas(pd.DataFrame(ood_incorrect))

    ood_hexshas = set(ood_incorrect["hexsha"])
    
    def _keep_condition(x):
        return x["hexsha"] not in ood_hexshas
    
    correct = correct.filter(_keep_condition, desc="Filtering ood")
    incorrect = incorrect.filter(_keep_condition, desc="Filtering ood")
    
    return correct, incorrect, ood_incorrect


def get_ood(correct, incorrect,args, ood_fn = make_source_program_ood):
    """
    Applies an OOD function to correct and incorrect prompts.
    Filters correct and incorrect prompts based on OOD and matching pairs.
    """
    if args.test_size > 0:
        correct, incorrect, ood = ood_fn(correct, incorrect,args)
    else:
        ood = None
        
    # does correct need same exact programs as incorrect?
    if args.do_fit_matching_pairs:
        intersection = set(incorrect["hexsha"]).intersection(set(correct["hexsha"]))
        incorrect = incorrect.filter(lambda x : x["hexsha"] in intersection, num_proc=cpu_count(), desc="Filtering ood")
        correct = correct.filter(lambda x : x["hexsha"] in intersection, num_proc=cpu_count(), desc="Filtering ood")
        assert set(incorrect["hexsha"]) == set(correct["hexsha"])
        
    return correct, incorrect, ood

                         
def fit_test_split_completions(dataset : datasets.Dataset, tokenizer, args):
    """
    For CAA and Completions datasets (equivalent)
    """
    if "generated_text" in dataset.column_names:
        # recompute "correct" sanity check: generated text starts with first token of fim_type
        dataset = dataset.map(lambda x : {"correct" : x["generated_text"].strip().startswith(
            tokenizer.decode(tokenizer.encode(x["fim_type"])[0])), **x}, desc="Sanity Recomputing correct")
        
    correct = dataset.filter(lambda x : x["correct"] == True, desc="Getting correct subset")
    incorrect = dataset.filter(lambda x : x["correct"] == False, desc="Getting incorrect subset")
    return get_ood(correct, incorrect, args)
    
def fit_test_split(dataset : datasets.Dataset, tokenizer, args):
    """
    For mutated datasets
    """
    correct = keep_columns(dataset, ["fim_program", "fim_type", "hexsha"])
    incorrect = keep_columns(dataset, ["mutated_program","fim_type","hexsha"])
    incorrect = incorrect.rename_columns({"mutated_program": "fim_program"})
    
    return get_ood(correct, incorrect, args)


def steer_on_ds(model, diff_tensor, incorrect, is_ood, args):
    """
    Given model, a steering tensor, and a dataset of incorrect prompts, steer on the dataset.
    Logs results to args.outdir
    """
    if is_ood:
        ood_flag = "ood_"
    else:
        ood_flag = ""
        
    if os.path.exists(f"{args.expdir}/{ood_flag}steering_results_ds"):
        print(f"...Loading steering results from {args.expdir}/{ood_flag}steering_results_ds...")
        steering_ds = datasets.load_from_disk(f"{args.expdir}/{ood_flag}steering_results_ds")
    else:
        print(f"...Applying patch to {ood_flag}incorrect prompts...")
        incorrect = datasets.Dataset.from_pandas(pd.DataFrame(incorrect))
        args.steering_outfile = f"{args.expdir}/{ood_flag}steering_predictions.json"
        steering_ds = steer(model,
                            incorrect,
                            diff_tensor, 
                            args)
        steering_ds.save_to_disk(f"{args.expdir}/{ood_flag}steering_results_ds")
        # remove cache files
        os.remove(args.steering_outfile)
        
    df = steering_ds.to_pandas()
    df = df[["steered_generation","fim_type","correct_steer", "hexsha"]]
    # sort by fim_type
    df = df.sort_values(by="fim_type")
    df.to_csv(f"{args.expdir}/{ood_flag}steering_results.csv")
    
    log_results(steering_ds, args, f"{ood_flag}eval_readme.md")


def steer(
    model,
    incorrect_eval,
    diff_tensor,
    args
):
    """
    Given a model, a steering tensor, and a dataset of incorrect prompts, steer on the dataset.
    
    """
    if "<FILL>" in incorrect_eval["fim_program"][0]:
        eval_prompts = [placeholder_to_std_fmt(ex["fim_program"], STARCODER_FIM) for ex in incorrect_eval]
    else:
        eval_prompts = [ex["fim_program"] for ex in incorrect_eval]
    eval_solutions = [ex["fim_type"] for ex in incorrect_eval]
        
    if args.custom_decoder is not False:
        decoder = torch.load(args.custom_decoder)
        weight = decoder["weight"]
        linear_layer = torch.nn.Linear(in_features=weight.shape[1], out_features=weight.shape[0])
        linear_layer.weight.data = weight
        custom_decoder = linear_layer.to("cpu")
    else:
        custom_decoder = None
        
    predictions = batched_insert_patch_logit(model, 
                eval_prompts, 
                diff_tensor, 
                args.layers_to_patch,
                args.tokens_to_patch,
                args.patch_mode,
                args.batch_size,
                args.steering_outfile,
                solutions=eval_solutions,
                custom_decoder=custom_decoder)
    steering_results = []
    for i,tok in enumerate(predictions):
        ex = incorrect_eval[i]
        steering_results.append({"steered_generation" : tok, 
                            # tok is always one token, fim_type len may vary
                            "correct_steer" : (ex["fim_type"].startswith(tok) and len(tok) > 0),
                            **ex})
            
    steering_ds = datasets.Dataset.from_pandas(pd.DataFrame(steering_results))
    return steering_ds


def get_steering_tensor(model, correct, incorrect, args):
    """
    Given a model, a dataset of correct and incorrect prompts, and args, return a steering tensor
    from the average of the incorrect prompts and the average of the correct prompts.
    """
    # load steering tensor if it exists, else create it
    if os.path.exists(f"{args.datadir}/steering_tensor.pt"):
        print(f"Loading steering tensor from {args.datadir}/steering_tensor.pt...")
        diff_tensor = torch.load(f"{args.datadir}/steering_tensor.pt")
        print(f"Diff tensor shape: {diff_tensor.shape}")
    else:
        if args.fim_placeholder:
            correct_prompts = [placeholder_to_std_fmt(ex["fim_program"], STARCODER_FIM) for ex in correct]
            incorrect_prompts = [placeholder_to_std_fmt(ex["fim_program"], STARCODER_FIM) for ex in incorrect]
        else:
            correct_prompts = [ex["fim_program"] for ex in correct]
            incorrect_prompts = [ex["fim_program"] for ex in incorrect]
            
        if os.path.exists(f"{args.datadir}/correct_avg_tensor.pt"):
            print(f"Loading correct avg tensor from {args.datadir}/correct_avg_tensor.pt...")
            correct_avg_tensor = torch.load(f"{args.datadir}/correct_avg_tensor.pt")
        else:
            print(f"Creating correct avg tensor...")
            correct_avg_tensor = batched_get_averages(model, 
                                                    correct_prompts,
                                                    args.tokens_to_patch,
                                                    batch_size=args.batch_size,
                                                    outfile=f"{args.datadir}/correct_avg_tensor")
            # save tensor
            torch.save(correct_avg_tensor, f"{args.datadir}/correct_avg_tensor.pt")
            # remove cache files
            os.remove(f"{args.datadir}/correct_avg_tensor.pkl")
            os.remove(f"{args.datadir}/correct_avg_tensor.json")
            
        if os.path.exists(f"{args.datadir}/incorrect_avg_tensor.pt"):
            print(f"Loading incorrect avg tensor from {args.datadir}/incorrect_avg_tensor.pt...")
            incorrect_avg_tensor = torch.load(f"{args.datadir}/incorrect_avg_tensor.pt")
        else:
            print(f"Creating incorrect avg tensor...")
            incorrect_avg_tensor = batched_get_averages(model,
                                                        incorrect_prompts,
                                                        args.tokens_to_patch,
                                                        batch_size=args.batch_size,
                                                        outfile=f"{args.datadir}/incorrect_avg_tensor")
            torch.save(incorrect_avg_tensor, f"{args.datadir}/incorrect_avg_tensor.pt")
            # remove cache files
            os.remove(f"{args.datadir}/incorrect_avg_tensor.pkl")
            os.remove(f"{args.datadir}/incorrect_avg_tensor.json")
            
        diff_tensor = correct_avg_tensor - incorrect_avg_tensor
        diff_tensor = rearrange(diff_tensor, "l t d -> l 1 t d") # [n_layers, n_prompts, n_tokens, n_embd]

        print(f"Diff tensor shape after transform: {diff_tensor.shape}")

        torch.save(diff_tensor, f"{args.datadir}/steering_tensor.pt")
        
    return diff_tensor

def get_data_info(ds, name) -> json:
    """
    Give some information about how balanced the ds is
    """
    if ds is None:
        return {}
    count_hex = Counter(ds["hexsha"])
    count_type = Counter(ds["fim_type"])
    len_ds = len(ds)
    return {"name":name, "length" : len_ds, "hexsha_counts" : count_hex, "type_counts" : count_type}


def log_results(steering_ds,  args, outfile):
    correct_steer = steering_ds.filter(lambda x : x["correct_steer"] == True, desc="Counting correct steers")
    metric = f"{len(correct_steer)} / {len(steering_ds)} = {len(correct_steer) / len(steering_ds)}"
    print(metric)
    steering_df_by_type = accuracy_per_type(steering_ds)
    
    with open(f"{args.expdir}/{outfile}", "w") as f:
        f.write(f"## Steering Results\n")
        f.write(metric)
        f.write("\n\n## Steering Results by Type\n\n")
        f.write(steering_df_by_type.to_markdown())
        

def accuracy_per_type(steering_ds):
    """
    Given a dataset of completions after steering, calculate accuracy per type.
    """
    # calculate accuracy per type
    steering_df = steering_ds.to_pandas()
    # cast correct_steer to int (0 if False, 1 if True)
    steering_df["correct_steer"] = steering_df["correct_steer"].astype(int)
    steering_df_by_type = steering_df.groupby("fim_type")
    steering_df_by_type = steering_df_by_type.agg({"correct_steer" : ["sum", "count"]}).reset_index()
    # make df of accuracy per type
    steering_df_by_type["accuracy"] = (steering_df_by_type[("correct_steer", "sum")] / 
                                       steering_df_by_type[("correct_steer", "count")])
    # normalize groups
    steering_df_by_type = steering_df_by_type.sort_values(by="accuracy", ascending=False)
    return steering_df_by_type.reset_index()
