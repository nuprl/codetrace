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


# def _sort_by_field(incorrect, field, reverse):
#     field_counts = Counter(incorrect[field])
#     sorted_field = sorted(field_counts.keys(), key= lambda x : field_counts[x], reverse=reverse)
#     return sorted_field

# def _make_ood(correct, incorrect,args):
#     """
#     OOD should have different set of programs and types.
#     """
#     ood_incorrect_a = _get_ood_subset(incorrect, "hexsha", args.test_size, reverse=False)
#     ood_incorrect_b = _get_ood_subset(incorrect, "fim_type", args.test_size, reverse=False)
    
#     ood_incorrect_a = incorrect.filter(lambda x : x["hexsha"] in _sort_by_field(incorrect, "hexsha", reverse=True))
#     ood_incorrect_b = incorrect.filter(lambda x : x["fim_type"] in _sort_by_field(incorrect, "fim_type", reverse=True))
    
#     # collect one from each ood until test_size is achieved
#     # but sort first
#     # hex_counts = Counter(incorrect["hexsha"])
#     # type_counts = Counter(incorrect["fim_type"])
#     # ood_incorrect_a = sorted(ood_incorrect_a, key=lambda x : hex_counts[x["hexsha"]], reverse=False)
#     # ood_incorrect_b = sorted(ood_incorrect_b, key=lambda x : type_counts[x["fim_type"]], reverse=True)
#     def _collect_condition(x, hex_example, type_example):
#         return x["hexsha"] == hex_example["hexsha"] or x["fim_type"] == type_example["fim_type"]

#     def _dedup_programs(x):
#         xcounts = Counter([k["fim_program"] for k in x])
#         return [k for k in x if xcounts[k["fim_program"]] == 1]
        
#     correct_acc = []
#     incorrect_acc = []
#     fit_threshold = 5
#     disable_progress_bar()
#     for i in tqdm(zip(ood_incorrect_a, ood_incorrect_b), desc="Collect fit and ood", total=len(ood_incorrect_a)):
        
#         if len(correct_acc) < fit_threshold:
#             correct_acc += correct.filter(lambda x : _collect_condition(x, i[0], i[1]))
#             correct_acc = _dedup_programs(correct_acc)
#         if len(incorrect_acc) < fit_threshold:
#             incorrect_acc += incorrect.filter(lambda x : _collect_condition(x, i[0], i[1]))
#             incorrect_acc = _dedup_programs(incorrect_acc)
        
#         if len(correct_acc) >= fit_threshold and len(incorrect_acc) >= fit_threshold:
#             break
#     enable_progress_bar()
#     # save
#     with open(f"{args.outdir}/incorrect_acc.json", "w") as f:
#         json.dump(incorrect_acc, f, indent=4)
#     with open(f"{args.outdir}/correct_acc.json", "w") as f:
#         json.dump(correct_acc, f, indent=4)
        
#     correct = datasets.Dataset.from_pandas(pd.DataFrame(correct_acc))
#     incorrect = datasets.Dataset.from_pandas(pd.DataFrame(incorrect_acc))
    
#     fit_hexshas = set(correct["hexsha"]).union(set(incorrect["hexsha"]))
#     fit_types = set(correct["fim_type"]).union(set(incorrect["fim_type"]))
#     ood_incorrect = incorrect.filter(lambda x : x["hexsha"] not in fit_hexshas and x["fim_type"] not in fit_types)
#     ood_incorrect = datasets.Dataset.from_pandas(pd.DataFrame(ood_incorrect))
    
#     # does correct need same exact programs as incorrect?
#     if args.do_fit_matching_pairs:
#         intersection = set(correct["hexsha"]).intersection(set(incorrect["hexsha"]))
#         correct = correct.filter(lambda x : x["hexsha"] in intersection, num_proc=cpu_count())
#         incorrect = incorrect.filter(lambda x : x["hexsha"] in intersection, num_proc=cpu_count())
    
#     return correct, incorrect, ood_incorrect

def keep_columns(ds, cols):
    columns = [c for c in ds.column_names if c not in cols]
    return ds.remove_columns(columns)

def _get_ood_subset(incorrect, field, test_len, reverse):
     # set aside some incorrect prompts
    field_counts = Counter(incorrect[field])
    # accumulate from the least count until threshold test_size is achieved
    sorted_field = sorted(list(incorrect), key= lambda x : field_counts[x[field]], reverse=reverse)
    ood_subset = []
    for x in sorted_field:
        ood_subset.append(x)
        if len(ood_subset) > test_len:
            break
    return ood_subset

# def _get_field_ood(correct, incorrect, field, test_len, reverse):
#     ood_subset = _get_ood_subset(incorrect, field, test_len, reverse)
#     incorrect_ood = incorrect.filter(lambda x : x[field] in ood_subset)
#     incorrect = incorrect.filter(lambda x : x[field] not in ood_subset)
#     correct = correct.filter(lambda x : x[field] not in ood_subset)
#     return correct, incorrect, incorrect_ood

def _dedup_programs(x):
    xcounts = Counter([k["fim_program"] for k in x])
    return [k for k in x if xcounts[k["fim_program"]] == 1]

def _make_ood(correct, incorrect,args):
    """
    OOD should have different set of programs and types.
    """
    if args.test_size > 1:
        test_len = args.test_size
    else:
        test_len = int(len(incorrect) * args.test_size)
        
    ood_incorrect_a = _get_ood_subset(incorrect, "hexsha", test_len, reverse=True)
    ood_incorrect_b = _get_ood_subset(incorrect, "fim_type", test_len, reverse=True)
    
    ood_incorrect = []
    for x in zip(ood_incorrect_a, ood_incorrect_b):
        ood_incorrect.append(x[0])
        ood_incorrect.append(x[1])
        ood_incorrect = _dedup_programs(ood_incorrect)
        if len(ood_incorrect) > test_len:
            break
    
    ood_incorrect = datasets.Dataset.from_pandas(pd.DataFrame(ood_incorrect))
    ood_types = set(ood_incorrect["fim_type"])
    ood_hexshas = set(ood_incorrect["hexsha"])
    
    def _keep_condition(x):
        return x["hexsha"] not in ood_hexshas and x["fim_type"] not in ood_types
    
    correct = correct.filter(_keep_condition)
    incorrect = incorrect.filter(_keep_condition)
    
    # does correct need same exact programs as incorrect?
    if args.do_fit_matching_pairs:
        intersection = set(correct["hexsha"]).intersection(set(incorrect["hexsha"]))
        correct = correct.filter(lambda x : x["hexsha"] in intersection, num_proc=cpu_count())
        incorrect = incorrect.filter(lambda x : x["hexsha"] in intersection, num_proc=cpu_count())
    
    return correct, incorrect, ood_incorrect


def fit_test_split_completions(dataset : datasets.Dataset, tokenizer, args):
    if "generated_text" in dataset.column_names:
        # recompute "correct": generated text starts with first token of fim_type
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
    
    intersection = set(correct["hexsha"]).intersection(set(incorrect["hexsha"]))
    correct = correct.filter(lambda x : x["hexsha"] in intersection, num_proc=cpu_count())
    incorrect = incorrect.filter(lambda x : x["hexsha"] in intersection, num_proc=cpu_count())
    
    if args.test_size > 0:
        return _make_ood(correct, incorrect,args)
    else:
        return correct, incorrect, None
    
def fit_test_split(dataset : datasets.Dataset, tokenizer, args):
    if args.prog_threshold > 0 or args.type_threshold > 0:
        dataset = filter_prompts(dataset, 
                                dedup_prog_threshold=args.prog_threshold, 
                                dedup_type_threshold=args.type_threshold)
    correct = keep_columns(dataset, ["fim_program", "fim_type", "hexsha"])
    incorrect = keep_columns(dataset, ["renamed_fim_program","fim_type","hexsha"])
    incorrect = incorrect.rename_columns({"renamed_fim_program": "fim_program"})
    if args.test_size > 0:
        return _make_ood(correct, incorrect,args)
    else:
        return correct, incorrect, None

def _pretty_info(ds) -> str:
    """
    Give some information about how balanced the ds is
    """
    if ds is None:
        return "None dataset"
    count_hex = Counter(ds["hexsha"])
    count_type = Counter(ds["fim_type"])
    len_ds = len(ds)
    return f"Length: {len_ds}\nHexsha counts: {count_hex}\nType counts: {count_type}\n"

def _steer_on_ds(model, diff_tensor, incorrect, is_ood, args):
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
        df = steering_ds.to_pandas()
        df = df[["steered_generation","fim_type","correct_steer", "hexsha"]]
        df.to_csv(f"{args.outdir}/{ood_flag}steering_results.csv")
        # remove cache files
        os.remove(args.steering_outfile)
            
    _log_results(steering_ds, args, f"{ood_flag}eval_readme.md")


def steer(
    model,
    incorrect_eval,
    diff_tensor,
    args
):
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

def _log_results(steering_ds,  args, outfile):
    correct_steer = steering_ds.filter(lambda x : x["correct_steer"] == True)
    metric = f"{len(correct_steer)} / {len(steering_ds)} = {len(correct_steer) / len(steering_ds)}"
    print(metric)
    with open(f"{args.outdir}/{outfile}", "w") as f:
        f.write(f"## Steering Results\n")
        f.write(metric)
        # write arguments of parser
        f.write(f"\n## Arguments\n")
        parser = vars(args)
        for k,v in parser.items():
            f.write(f"{k} : {v}\n")

def _get_steering_tensor(model, correct, incorrect, args):
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
    
def main():
    # ==========================================================================================
    # PART 0: setup
    # ==========================================================================================
    steering_args = sys.argv[1]
    with open(steering_args, "r") as f:
        args = json.load(f)
    args = Namespace(**args)

    # parent dir
    exp_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    args.outdir = f"{exp_dir}/exp_data/{args.outdir}"
    os.makedirs(args.outdir, exist_ok=True)
    
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
    # PART 1: filter
    # ==========================================================================================
    print("...Generating fit test split...")
    
    if "completions" in args.dataset:
        correct, incorrect, incorrect_eval = fit_test_split_completions(ds,model.tokenizer, args)
    else:
        correct, incorrect, incorrect_eval = fit_test_split(ds,model.tokenizer, args)
        
    if args.max_size > 0:
        correct = correct.select(range(args.max_size))
        incorrect = incorrect.select(range(args.max_size))
        
    print("Correct, incorrect, incorrect ood len:", len(correct), len(incorrect), len(incorrect_eval))
    info_incorrect = _pretty_info(incorrect)
    info_correct = _pretty_info(correct)
    info_eval = _pretty_info(incorrect_eval)

    with open(f"{args.outdir}/data_readme.md", "w") as f:
        info = f"Correct\n{info_correct}\n\nIncorrect\n{info_incorrect}\n\nIncorrect Eval\n{info_eval}\n"
        f.write(info)

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