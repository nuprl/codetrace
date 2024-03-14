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

def _dedup_ds(ds, key):
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


def _get_field_subset(incorrect, field, test_len, reverse):
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


def _make_source_program_ood(correct, incorrect,args):
    """
    OOD should have different set of source programs
    """
    if args.test_size > 1:
        test_len = args.test_size
    else:
        test_len = int(len(incorrect) * args.test_size)
        
    ood_incorrect = _get_field_subset(incorrect, "hexsha", test_len, reverse=False)
    ood_incorrect = datasets.Dataset.from_pandas(pd.DataFrame(ood_incorrect))

    ood_hexshas = set(ood_incorrect["hexsha"])
    
    def _keep_condition(x):
        return x["hexsha"] not in ood_hexshas
    
    correct = correct.filter(_keep_condition)
    incorrect = incorrect.filter(_keep_condition)
    
    return correct, incorrect, ood_incorrect


def _get_ood(correct, incorrect,args, ood_fn = _make_source_program_ood):
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
        incorrect = incorrect.filter(lambda x : x["hexsha"] in intersection, num_proc=cpu_count())
        correct = correct.filter(lambda x : x["hexsha"] in intersection, num_proc=cpu_count())
        assert set(incorrect["hexsha"]) == set(correct["hexsha"])
        
    return correct, incorrect, ood

                         
def fit_test_split_completions(dataset : datasets.Dataset, tokenizer, args):
    """
    For CAA and Completions datasets (equivalent)
    """
    if "generated_text" in dataset.column_names:
        # recompute "correct" sanity check: generated text starts with first token of fim_type
        dataset = dataset.map(lambda x : {"correct" : x["generated_text"].strip().startswith(
            tokenizer.decode(tokenizer.encode(x["fim_type"])[0])), **x})
        
    correct = dataset.filter(lambda x : x["correct"] == True)
    incorrect = dataset.filter(lambda x : x["correct"] == False)
    if args.prog_threshold > 0 or args.type_threshold > 0:
        correct = filter_prompts(correct, 
                                dedup_prog_threshold=args.prog_threshold, 
                                dedup_type_threshold=args.type_threshold)
        incorrect = filter_prompts(incorrect,
                                    dedup_prog_threshold=args.prog_threshold, 
                                    dedup_type_threshold=args.type_threshold)
    return _get_ood(correct, incorrect, args)
    
def fit_test_split(dataset : datasets.Dataset, tokenizer, args):
    """
    For renamed datasets
    """
    if args.prog_threshold > 0 or args.type_threshold > 0:
        dataset = filter_prompts(dataset, 
                                dedup_prog_threshold=args.prog_threshold, 
                                dedup_type_threshold=args.type_threshold)
    correct = keep_columns(dataset, ["fim_program", "fim_type", "hexsha"])
    incorrect = keep_columns(dataset, ["renamed_fim_program","fim_type","hexsha"])
    incorrect = incorrect.rename_columns({"renamed_fim_program": "fim_program"})
    
    return _get_ood(correct, incorrect, args)


def _steer_on_ds(model, diff_tensor, incorrect, is_ood, args):
    """
    Given model, a steering tensor, and a dataset of incorrect prompts, steer on the dataset.
    Logs results to args.outdir
    """
    if is_ood:
        ood_flag = "ood_"
    else:
        ood_flag = ""
        
    if os.path.exists(f"{args.outdir}/{ood_flag}steering_results_ds"):
        print(f"...Loading steering results from {args.outdir}/{ood_flag}steering_results_ds...")
        steering_ds = datasets.load_from_disk(f"{args.outdir}/{ood_flag}steering_results_ds")
    else:
        print(f"...Applying patch to {ood_flag}incorrect prompts...")
        incorrect = datasets.Dataset.from_pandas(pd.DataFrame(incorrect))
        args.steering_outfile = f"{args.outdir}/{ood_flag}steering_predictions.json"
        steering_ds = steer(model,
                            incorrect,
                            diff_tensor, 
                            args)
        steering_ds.save_to_disk(f"{args.outdir}/{ood_flag}steering_results_ds")
        # remove cache files
        os.remove(args.steering_outfile)
        
    df = steering_ds.to_pandas()
    df = df[["steered_generation","fim_type","correct_steer", "hexsha"]]
    # sort by fim_type
    df = df.sort_values(by="fim_type")
    df.to_csv(f"{args.outdir}/{ood_flag}steering_results.csv")
    
    _log_results(steering_ds, args, f"{ood_flag}eval_readme.md")


def steer(
    model,
    incorrect_eval,
    diff_tensor,
    args
):
    """
    Given a model, a steering tensor, and a dataset of incorrect prompts, steer on the dataset.
    
    """
    if args.fim_placeholder:
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



def _get_steering_tensor(model, correct, incorrect, args):
    """
    Given a model, a dataset of correct and incorrect prompts, and args, return a steering tensor
    from the average of the incorrect prompts and the average of the correct prompts.
    """
    # load steering tensor if it exists, else create it
    if os.path.exists(f"{args.outdir}/steering_tensor.pt"):
        print(f"...Loading steering tensor from {args.outdir}/steering_tensor.pt...")
        diff_tensor = torch.load(f"{args.outdir}/steering_tensor.pt")
        print(f"Diff tensor shape: {diff_tensor.shape}")
    else:
        if args.fim_placeholder:
            correct_prompts = [placeholder_to_std_fmt(ex["fim_program"], STARCODER_FIM) for ex in correct]
            incorrect_prompts = [placeholder_to_std_fmt(ex["fim_program"], STARCODER_FIM) for ex in incorrect]
        else:
            correct_prompts = [ex["fim_program"] for ex in correct]
            incorrect_prompts = [ex["fim_program"] for ex in incorrect]
            
        if os.path.exists(f"{args.outdir}/correct_avg_tensor.pt"):
            print(f"...Loading correct avg tensor from {args.outdir}/correct_avg_tensor.pt...")
            correct_avg_tensor = torch.load(f"{args.outdir}/correct_avg_tensor.pt")
        else:
            print(f"...Creating correct avg tensor...")
            correct_avg_tensor = batched_get_averages(model, 
                                                    correct_prompts,
                                                    args.tokens_to_patch,
                                                    batch_size=args.batch_size,
                                                    outfile=f"{args.outdir}/correct_avg_tensor")
            # save tensor
            torch.save(correct_avg_tensor, f"{args.outdir}/correct_avg_tensor.pt")
            # remove cache files
            os.remove(f"{args.outdir}/correct_avg_tensor.pkl")
            os.remove(f"{args.outdir}/correct_avg_tensor.json")
            
        if os.path.exists(f"{args.outdir}/incorrect_avg_tensor.pt"):
            print(f"...Loading incorrect avg tensor from {args.outdir}/incorrect_avg_tensor.pt...")
            incorrect_avg_tensor = torch.load(f"{args.outdir}/incorrect_avg_tensor.pt")
        else:
            print(f"...Creating incorrect avg tensor...")
            incorrect_avg_tensor = batched_get_averages(model,
                                                        incorrect_prompts,
                                                        args.tokens_to_patch,
                                                        batch_size=args.batch_size,
                                                        outfile=f"{args.outdir}/incorrect_avg_tensor")
            torch.save(incorrect_avg_tensor, f"{args.outdir}/incorrect_avg_tensor.pt")
            # remove cache files
            os.remove(f"{args.outdir}/incorrect_avg_tensor.pkl")
            os.remove(f"{args.outdir}/incorrect_avg_tensor.json")
            
        diff_tensor = correct_avg_tensor - incorrect_avg_tensor
        diff_tensor = rearrange(diff_tensor, "l t d -> l 1 t d") # [n_layers, n_prompts, n_tokens, n_embd]

        print(f"Diff tensor shape after transform: {diff_tensor.shape}")

        torch.save(diff_tensor, f"{args.outdir}/steering_tensor.pt")
        
    return diff_tensor

def _get_data_info(ds, name) -> json:
    """
    Give some information about how balanced the ds is
    """
    if ds is None:
        return {}
    count_hex = Counter(ds["hexsha"])
    count_type = Counter(ds["fim_type"])
    len_ds = len(ds)
    return {"name":name, "length" : len_ds, "hexsha_counts" : count_hex, "type_counts" : count_type}


def _log_results(steering_ds,  args, outfile):
    correct_steer = steering_ds.filter(lambda x : x["correct_steer"] == True)
    metric = f"{len(correct_steer)} / {len(steering_ds)} = {len(correct_steer) / len(steering_ds)}"
    print(metric)
    steering_df_by_type = _accuracy_per_type(steering_ds)
    
    with open(f"{args.outdir}/{outfile}", "w") as f:
        f.write(f"## Steering Results\n")
        f.write(metric)
        f.write("\n\n## Steering Results by Type\n\n")
        f.write(steering_df_by_type.to_markdown())
        

def _accuracy_per_type(steering_ds):
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

def main():
    # ==========================================================================================
    # PART 0: setup
    # ==========================================================================================
    steering_args = sys.argv[1]
    with open(steering_args, "r") as f:
        args = json.load(f)
    args = Namespace(**args)

    # parent dir
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    experiment_dir = f"{parent_dir}/{args.outdir}"
    os.makedirs(experiment_dir, exist_ok=True)
    
    # make a unique run_id for given args
    if args.run_name == False:
        run_id = f"run_{hashlib.sha256(str(args).encode()).hexdigest()}"
    else:
        run_id = args.run_name
        
    outdir = f"{experiment_dir}/{run_id}"
    os.makedirs(outdir, exist_ok=True)
    with open(f"{outdir}/args_steering.json", "w") as f:
        json.dump(vars(args), f, indent=4)
    args.outdir = outdir
    
    ds = datasets.load_dataset(args.dataset, split="train")
    print(ds)
    
    if args.shuffle:
        ds = ds.shuffle(seed=42)
        
    # filter out too large prompts for OOM
    ds = ds.filter(lambda x : len(x["fim_program"]) < 8000)
    
    if "<FILL>" in ds["fim_program"][0]:
        args.fim_placeholder = True
    else:
        args.fim_placeholder = False
        
    model = LanguageModel(args.model, device_map="cuda")

    # ==========================================================================================
    # PART 1: filter data
    # ==========================================================================================
    print("...Generating fit test split...")
    
    if "completions" in args.dataset or "caa" in args.dataset:
        correct, incorrect, incorrect_eval = fit_test_split_completions(ds,model.tokenizer, args)
    else:
        correct, incorrect, incorrect_eval = fit_test_split(ds,model.tokenizer, args)
        
    if args.max_size > 0:
        correct = correct.select(range(args.max_size))
        incorrect = incorrect.select(range(args.max_size))
        incorrect_eval = incorrect_eval.select(range(args.max_size))
        
    print("Correct, incorrect, incorrect ood len:", len(correct), len(incorrect), len(incorrect_eval))
    info_incorrect = _get_data_info(incorrect, "incorrect")
    info_correct = _get_data_info(correct, "correct")
    info_ood = _get_data_info(incorrect_eval, "incorrect_ood")

    with open(f"{args.outdir}/data_info.json", "w") as f:
        info = [info_incorrect, info_correct, info_ood]
        # add types in info_eval that are not in info_incorrect
        if len(info_ood) > 0:
            types = list(set(info_ood["type_counts"].keys()).difference(set(info_incorrect["type_counts"].keys())))
            ood_types = {t : int(info_ood["type_counts"][t]) for t in types}
            info.append({"ood_types" : ood_types})
        json.dump(info, f, indent=4)


    # ==========================================================================================
    # PART 2: averages
    # ==========================================================================================
    
    diff_tensor = _get_steering_tensor(model, correct, incorrect, args)

    #==========================================================================================
    # PART 3: steering on train
    #==========================================================================================
    
    _steer_on_ds(model, diff_tensor, incorrect, False, args)
        
    # ==========================================================================================
    # PART 4: steering ood
    # ==========================================================================================
    if args.test_size > 0:
        _steer_on_ds(model, diff_tensor, incorrect_eval, True, args)
    
    
if __name__ == "__main__":
    main()