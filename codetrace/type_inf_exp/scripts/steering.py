from codetrace.type_inf_exp.request_patch import *
from codetrace.utils import *
from einops import rearrange
from argparse import ArgumentParser, Namespace
from collections import Counter

def keep_columns(ds, cols):
    columns = [c for c in ds.column_names if c not in cols]
    return ds.remove_columns(columns)

# def fit_test_split(dataset : datasets.Dataset, tokenizer, args):
#     dataset = filter_prompts(dataset, 
#                              single_tokenize=tokenizer, 
#                              dedup_prog_threshold=args.prog_threshold, 
#                              dedup_type_threshold=args.type_threshold)
    
#     if args.test_size > 0:
#         dataset = dataset.shuffle(seed=42)
#         test_size = int(len(dataset) * args.test_size)
#         incorrect_eval = datasets.Dataset.from_pandas(pd.DataFrame(dataset[:test_size]))
#         dataset = datasets.Dataset.from_pandas(pd.DataFrame(dataset[test_size:]))
    
#     correct = keep_columns(dataset, ["fim_program", "fim_type", "hexsha"])
#     incorrect = keep_columns(dataset, ["renamed_fim_program","fim_type","hexsha","renamed_generated_text","renamed_variables","renamed_percent"])
#     incorrect = incorrect.rename_columns({"renamed_fim_program": "fim_program", "renamed_generated_text":"generated_text"})
    
#     if args.test_size > 0:
#         incorrect_eval = keep_columns(incorrect_eval, ["renamed_fim_program","fim_type","hexsha","renamed_generated_text","renamed_variables","renamed_percent"])
#         incorrect_eval = incorrect_eval.rename_columns({"renamed_fim_program": "fim_program", "renamed_generated_text":"generated_text"})
#         return correct, incorrect, incorrect_eval
#     else:
#         return correct, incorrect, None

def fit_test_split(dataset : datasets.Dataset, tokenizer, args):
    dataset = filter_prompts(dataset, 
                             single_tokenize=tokenizer, 
                             dedup_prog_threshold=args.prog_threshold, 
                             dedup_type_threshold=args.type_threshold)
    correct = keep_columns(dataset, ["fim_program", "fim_type", "hexsha"])
    incorrect = keep_columns(dataset, ["renamed_fim_program","fim_type","hexsha","renamed_generated_text","renamed_variables","renamed_percent"])
    incorrect = incorrect.rename_columns({"renamed_fim_program": "fim_program", "renamed_generated_text":"generated_text"})
    if args.test_size > 0:
        # set aside some incorrect prompts
        random.seed(4)
        hexshas = list(incorrect["hexsha"])
        hexshas = random.sample(hexshas, int(len(hexshas) * args.test_size))
        incorrect_eval = incorrect.filter(lambda x : x["hexsha"] in hexshas)
        incorrect = incorrect.filter(lambda x : x["hexsha"] not in hexshas)
        correct = correct.filter(lambda x : x["hexsha"] not in hexshas)
        return correct, incorrect, incorrect_eval
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
    args,
    incorrect_eval,
    diff_tensor,
    layers_to_patch,
    tokens_to_patch,
    patch_mode,
    batch_size
):
    eval_prompts = [placeholder_to_std_fmt(ex["fim_program"], STARCODER_FIM) for ex in incorrect_eval]
    # print types in incorrect
    
    out = batched_insert_patch(model, 
                eval_prompts, 
                diff_tensor, 
                layers_to_patch,
                tokens_to_patch,
                patch_mode,
                batch_size)
    steering_results = []
    for i,trace_res in tqdm(enumerate(out), desc="Logits"):
        prompt_len = trace_res._logits.shape[1]
        logits : LogitResult = trace_res.decode_logits(prompt_idx=list(range(prompt_len)), do_log_probs=False)

        for j in list(range(prompt_len)):
            tok = logits[-1][j][-1].tokens(model.tokenizer)
            assert len(tok) == 1, tok
            tok = tok[0].strip()
            ex = incorrect_eval[(i*args.batch_size)+j]
            steering_results.append({"steered_generation" : tok, 
                            "correct_steer" : (tok == ex["fim_type"]),
                            **ex})
    steering_ds = datasets.Dataset.from_pandas(pd.DataFrame(steering_results))
    return steering_ds

def main():
    # ==========================================================================================
    # PART 0: setup
    # ==========================================================================================
    steering_args = os.path.join(os.path.dirname(__file__), "args_steering.json")
    with open(steering_args, "r") as f:
        args = json.load(f)
    args = Namespace(**args)

    exp_dir = "/home/franlucc/projects/codetrace/codetrace/type_inf_exp"
    ds = datasets.load_dataset(args.dataset, split="train")
    
    # filter out too large prompts for OOM
    ds = ds.filter(lambda x : len(x["fim_program"]) < 8000 and x["fim_type"] not in ["this","{}"])

    model = LanguageModel(args.model, device_map="cuda")

    out_dir = f"{exp_dir}/exp_data/{args.outdir}"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # ==========================================================================================
    # PART 1: filter
    # ==========================================================================================
    print("...Generating fit test split...")
    
    correct, incorrect, incorrect_eval = fit_test_split(ds,model.tokenizer, args)

    info_incorrect = _pretty_print(incorrect)
    info_correct = _pretty_print(correct)
    if incorrect_eval is not None:
        info_eval = _pretty_print(incorrect_eval)
    else:
        info_eval = "No eval set"

    info = f"Correct\n{info_correct}\n\nIncorrect\n{info_incorrect}\n\nIncorrect Eval\n{info_eval}\n"

    with open(f"{out_dir}/data_readme.md", "w") as f:
        f.write(info)
        
    print(info)

    # # # ==========================================================================================
    # # # PART 2: averages
    # # # ==========================================================================================
    print(f"...Getting averages for correct and incorrect prompts...")

    correct_prompts = [placeholder_to_std_fmt(ex["fim_program"], STARCODER_FIM) for ex in correct]
    incorrect_prompts = [placeholder_to_std_fmt(ex["fim_program"], STARCODER_FIM) for ex in incorrect]
    correct_avg_tensor = batched_get_averages(model, 
                                            correct_prompts,
                                            args.tokens_to_patch,
                                                batch_size=args.batch_size)

    incorrect_avg_tensor = batched_get_averages(model,
                                                incorrect_prompts,
                                                args.tokens_to_patch,
                                                batch_size=args.batch_size)
                                            
    diff_tensor = correct_avg_tensor - incorrect_avg_tensor
    diff_tensor = rearrange(diff_tensor, "l t d -> l 1 t d") # [n_layers, n_prompts, n_tokens, n_embd]

    print(f"Diff tensor shape after transform: {diff_tensor.shape}")

    torch.save(diff_tensor, f"{out_dir}/steering_tensor.pt")

    #==========================================================================================
    # PART 3: steering on train
    #==========================================================================================
    
    def _evaluate(steering_ds, counts, args, out_dir, outfile):
        correct_steer = steering_ds.filter(lambda x : x["correct_steer"] == True)
        metric = f"{len(correct_steer)} / {len(steering_ds)} = {len(correct_steer) / len(steering_ds)}"
        print(metric)
        with open(f"{out_dir}/{outfile}", "w") as f:
            f.write(f"## Steering Results\n")
            f.write(metric)
            # write arguments of parser
            f.write(f"\n## Arguments\n")
            parser = vars(args)
            for k,v in parser.items():
                f.write(f"{k} : {v}\n")
            f.write("\nEval type distribution\n")
            f.write(str(counts))
            
    print(f"...Applying patch to incorrect prompts...")
    incorrect = datasets.Dataset.from_pandas(pd.DataFrame(incorrect))
    counts = Counter(incorrect["fim_type"])
    print(counts)

    steering_ds = steer(model, 
                        args,
                        incorrect,
                        diff_tensor,
                        args.layers_to_patch,
                        args.tokens_to_patch,
                        args.patch_mode,
                        args.batch_size)
    steering_ds.save_to_disk(f"{out_dir}/steering_results_ds")
    df = steering_ds.to_pandas()
    df = df[["steered_generation","fim_type","correct_steer","generated_text", "hexsha"]]
    df.to_csv(f"{out_dir}/steering_results.csv")
            
    _evaluate(steering_ds, counts, args, out_dir, "eval_readme.md")
        
    # # ==========================================================================================
    # # # PART 4: steering ood
    # # ==========================================================================================
    if args.test_size > 0:
        print(f"...Applying patch to incorrect prompts...")

        incorrect_eval = datasets.Dataset.from_pandas(pd.DataFrame(incorrect_eval))
        counts = Counter(incorrect["fim_type"])
        print(counts)

        steering_ds = steer(model, 
                            args,
                            incorrect_eval,
                            diff_tensor,
                            args.layers_to_patch,
                            args.tokens_to_patch,
                            args.patch_mode,
                            args.batch_size)
        steering_ds.save_to_disk(f"{out_dir}/steering_results_ood")
        df = steering_ds.to_pandas()
        df = df[["steered_generation","fim_type","correct_steer","generated_text", "hexsha"]]
        df.to_csv(f"{out_dir}/steering_results_ood.csv")
        
        _evaluate(steering_ds, counts, args, out_dir, "eval_ood_readme.md")
    
    
if __name__ == "__main__":
    main()