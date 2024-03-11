from codetrace.type_inf_exp.request_patch import *
from codetrace.utils import *
from einops import rearrange
from argparse import ArgumentParser, Namespace
from collections import Counter
import hashlib
import sys
import pickle
from copy import deepcopy

def keep_columns(ds, cols):
    columns = [c for c in ds.column_names if c not in cols]
    return ds.remove_columns(columns)

def fit_test_split_completions(dataset : datasets.Dataset, tokenizer, args):
    # recompute "correct"
    dataset = dataset.map(lambda x : {"correct" : x["generated_text"].strip().startswith(x["fim_type"].strip()), **x})
    correct = dataset.filter(lambda x : x["correct"] == True)
    incorrect = dataset.filter(lambda x : x["correct"] == False)
    correct = filter_prompts(correct, 
                             single_tokenize=tokenizer, 
                             dedup_prog_threshold=args.prog_threshold, 
                             dedup_type_threshold=args.type_threshold)
    incorrect = filter_prompts(incorrect,
                                 single_tokenize=tokenizer, 
                                 dedup_prog_threshold=args.prog_threshold, 
                                 dedup_type_threshold=args.type_threshold)
    # keep only hexshas in incorrect
    correct = correct.filter(lambda x : x["hexsha"] in set(incorrect["hexsha"]))
    incorrect = incorrect.filter(lambda x : x["hexsha"] in set(correct["hexsha"]))
    
    if args.test_size > 0:
        # set aside some incorrect prompts
        hexsha_counts = Counter(incorrect["hexsha"])
        # accumulate hexshas from the least count until threshold test_size is achieved
        sorted_hexshas = sorted(hexsha_counts.keys(), key= lambda x : hexsha_counts[x])
        test_len = int(len(incorrect) * args.test_size)
        ood_hexshas = []
        count = 0
        for hexsha in sorted_hexshas:
            ood_hexshas.append(hexsha)
            count += hexsha_counts[hexsha]
            if count > test_len:
                break
        incorrect_ood = incorrect.filter(lambda x : x["hexsha"] in ood_hexshas)
        incorrect = incorrect.filter(lambda x : x["hexsha"] not in ood_hexshas)
        correct = correct.filter(lambda x : x["hexsha"] not in ood_hexshas)
        return correct, incorrect, incorrect_ood
    else:
        return correct, incorrect, None
    
def fit_test_split(dataset : datasets.Dataset, tokenizer, args):
    dataset = filter_prompts(dataset, 
                             single_tokenize=tokenizer, 
                             dedup_prog_threshold=args.prog_threshold, 
                             dedup_type_threshold=args.type_threshold)
    correct = keep_columns(dataset, ["fim_program", "fim_type", "hexsha"])
    incorrect = keep_columns(dataset, ["renamed_fim_program","fim_type","hexsha"])
    incorrect = incorrect.rename_columns({"renamed_fim_program": "fim_program"})
    if args.test_size > 0:
        # set aside some incorrect prompts
        hexsha_counts = Counter(incorrect["hexsha"])
        # accumulate hexshas from the least count until threshold test_size is achieved
        sorted_hexshas = sorted(hexsha_counts.keys(), key= lambda x : hexsha_counts[x])
        test_len = int(len(incorrect) * args.test_size)
        ood_hexshas = []
        count = 0
        for hexsha in sorted_hexshas:
            ood_hexshas.append(hexsha)
            count += hexsha_counts[hexsha]
            if count > test_len:
                break
        incorrect_ood = incorrect.filter(lambda x : x["hexsha"] in ood_hexshas)
        incorrect = incorrect.filter(lambda x : x["hexsha"] not in ood_hexshas)
        correct = correct.filter(lambda x : x["hexsha"] not in ood_hexshas)
        return correct, incorrect, incorrect_ood
    else:
        return correct, incorrect, None

def _pretty_print(ds) -> str:
    """
    Give some information about how balanced the ds is
    """
    df = pd.DataFrame(ds)
    s = ""
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        s += str(df["fim_type"].value_counts())
        s += "\n"
        s += str(df["hexsha"].value_counts())
        s += "\n"
        s += str(len(df))
    return s

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
    # print types in incorrect
    
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
                            "correct_steer" : (ex["fim_type"].startswith(tok) and len(ex["fim_type"]) > 0),
                            **ex})
            
    steering_ds = datasets.Dataset.from_pandas(pd.DataFrame(steering_results))
    return steering_ds

def _evaluate(steering_ds, counts, args, outfile):
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
        f.write("\nEval type distribution\n")
        f.write(str(counts))

def _get_steering_tensor(model, correct, incorrect, args):
    # load steering tensor if it exists, else create it
    if os.path.exists(f"{args.outdir}/steering_tensor.pt"):
        print(f"...Loading steering tensor from {args.outdir}/steering_tensor.pt...")
        diff_tensor = torch.load(f"{args.outdir}/steering_tensor.pt")
        print(f"Diff tensor shape: {diff_tensor.shape}")
    else:
        print(f"...Creating steering tensor...")
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
            correct_avg_tensor = batched_get_averages(model, 
                                                    correct_prompts,
                                                    args.tokens_to_patch,
                                                    batch_size=args.batch_size,
                                                    outfile=f"{args.outdir}/correct_avg_tensor")
            # save tensor
            torch.save(correct_avg_tensor, f"{args.outdir}/correct_avg_tensor.pt")
            
        if os.path.exists(f"{args.outdir}/incorrect_avg_tensor.pt"):
            print(f"...Loading incorrect avg tensor from {args.outdir}/incorrect_avg_tensor.pt...")
            incorrect_avg_tensor = torch.load(f"{args.outdir}/incorrect_avg_tensor.pt")
        else:
            incorrect_avg_tensor = batched_get_averages(model,
                                                        incorrect_prompts,
                                                        args.tokens_to_patch,
                                                        batch_size=args.batch_size,
                                                        outfile=f"{args.outdir}/incorrect_avg_tensor")
            torch.save(incorrect_avg_tensor, f"{args.outdir}/incorrect_avg_tensor.pt")
            
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
    # filter out too large prompts for OOM
    ds = ds.filter(lambda x : len(x["fim_program"]) < 8000)
    if args.fim_placeholder:
        assert "<FILL>" in ds["fim_program"][0]
    else:
        assert "<FILL>" not in ds["fim_program"][0]
        
    model = LanguageModel(args.model, device_map="cuda")

    # ==========================================================================================
    # PART 1: filter
    # ==========================================================================================
    print("...Generating fit test split...")
    
    if "completions" in args.dataset:
        correct, incorrect, incorrect_eval = fit_test_split_completions(ds,model.tokenizer, args)
    else:
        correct, incorrect, incorrect_eval = fit_test_split(ds,model.tokenizer, args)

    print("Correct, incorrect, incorrect ood len:", len(correct), len(incorrect), len(incorrect_eval))
    info_incorrect = _pretty_print(incorrect)
    info_correct = _pretty_print(correct)
    if incorrect_eval is not None:
        info_eval = _pretty_print(incorrect_eval)
    else:
        info_eval = "No eval set"

    info = f"Correct\n{info_correct}\n\nIncorrect\n{info_incorrect}\n\nIncorrect Eval\n{info_eval}\n"

    with open(f"{args.outdir}/data_readme.md", "w") as f:
        f.write(info)
        
    # print(info)

    # ==========================================================================================
    # PART 2: averages
    # ==========================================================================================
    print(f"...Getting averages for correct and incorrect prompts...")
    
    diff_tensor = _get_steering_tensor(model, correct, incorrect, args)

    #==========================================================================================
    # PART 3: steering on train
    #==========================================================================================
            
    print(f"...Applying patch to incorrect prompts...")
    incorrect = datasets.Dataset.from_pandas(pd.DataFrame(incorrect))
    counts = Counter(incorrect["fim_type"])
    print(counts)
    args.steering_outfile = f"{args.outdir}/steered_predictions.json"
    steering_ds = steer(model,
                        incorrect,
                        diff_tensor, args)
    steering_ds.save_to_disk(f"{args.outdir}/steering_results_ds")
    df = steering_ds.to_pandas()
    df = df[["steered_generation","fim_type","correct_steer", "hexsha"]]
    df.to_csv(f"{args.outdir}/steering_results.csv")
            
    _evaluate(steering_ds, counts, args, "eval_readme.md")
        
    # ==========================================================================================
    # PART 4: steering ood
    # ==========================================================================================
    if args.test_size > 0:
        print(f"...Applying patch to incorrect OOD prompts...")
        args.steering_outfile = f"{args.outdir}/ood_steered_predictions.json"
        incorrect_eval = datasets.Dataset.from_pandas(pd.DataFrame(incorrect_eval))
        counts = Counter(incorrect["fim_type"])
        print(counts)

        steering_ds = steer(model, 
                            incorrect_eval,
                            diff_tensor,
                            args)
        steering_ds.save_to_disk(f"{args.outdir}/steering_results_ood")
        df = steering_ds.to_pandas()
        df = df[["steered_generation","fim_type","correct_steer", "hexsha"]]
        df.to_csv(f"{args.outdir}/steering_results_ood.csv")
        _evaluate(steering_ds, counts, args, "eval_ood_readme.md")
    
    
if __name__ == "__main__":
    main()