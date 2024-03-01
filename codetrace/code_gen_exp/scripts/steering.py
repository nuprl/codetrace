from codetrace.type_inf_exp.request_patch import *
from codetrace.utils import *
from einops import rearrange
from argparse import ArgumentParser, Namespace
from collections import Counter


def fit_test_split(dataset : datasets.Dataset, args):
    correct = dataset.remove_columns(["renamed_prompt","renamed_variables","renamed_percent","correct"]).rename_column("original_prompt","prompt")
    print(correct)
    incorrect = dataset.remove_columns(["original_prompt","correct"]).rename_column("renamed_prompt","prompt")
    print(incorrect)
    
    if args.test_size > 0:
        # set aside some incorrect prompts
        random.seed(4)
        hexshas = list(incorrect["name"])
        hexshas = random.sample(hexshas, int(len(hexshas) * args.test_size))
        incorrect_eval = incorrect.filter(lambda x : x["name"] in hexshas)
        incorrect = incorrect.filter(lambda x : x["name"] not in hexshas)
        correct = correct.filter(lambda x : x["name"] not in hexshas)
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
        s += str(df["name"].value_counts())
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
    """
    Need to steer with generation
    """
    pass
    # eval_prompts = ex["prompt"]
    # # print types in incorrect
    
    # out = batched_insert_patch(model, 
    #             eval_prompts, 
    #             diff_tensor, 
    #             layers_to_patch,
    #             tokens_to_patch,
    #             patch_mode,
    #             batch_size)
    # steering_results = []
    # for i,trace_res in tqdm(enumerate(out), desc="Logits"):
    #     prompt_len = trace_res._logits.shape[1]
    #     logits : LogitResult = trace_res.decode_logits(prompt_idx=list(range(prompt_len)), do_log_probs=False)

    #     for j in list(range(prompt_len)):
    #         tok = logits[-1][j][-1].tokens(model.tokenizer)
    #         assert len(tok) == 1, tok
    #         tok = tok[0].strip()
    #         ex = incorrect_eval[(i*args.batch_size)+j]
    #         steering_results.append({"steered_generation" : tok, 
    #                         **ex})
    # steering_ds = datasets.Dataset.from_pandas(pd.DataFrame(steering_results))
    # return steering_ds

def main():
    # ==========================================================================================
    # PART 0: setup
    # ==========================================================================================
    steering_args = os.path.join(os.path.dirname(__file__), "args_steering.json")
    with open(steering_args, "r") as f:
        args = json.load(f)
    args = Namespace(**args)

    exp_dir = "/home/franlucc/projects/codetrace/codetrace/codegen_gen_exp"
    ds = datasets.load_dataset(args.dataset, split="train")

    model = LanguageModel(args.model, device_map="cuda")

    out_dir = f"{exp_dir}/exp_data/{args.outdir}"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # ==========================================================================================
    # PART 1: filter
    # ==========================================================================================
    print("...Generating fit test split...")
    
    correct, incorrect, incorrect_eval = fit_test_split(ds, args)

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

    # ==========================================================================================
    # PART 2: averages 
    #  TODO: different ways we can extract avg
    # - last token [current]
    # - last token in FIM formulation
    # - all tokens
    # ==========================================================================================
    print(f"...Getting averages for correct and incorrect prompts...")
    
    # load steering tensor if it exists, else create it
    if os.path.exists(f"{out_dir}/steering_tensor.pt"):
        print(f"...Loading steering tensor from {out_dir}/steering_tensor.pt...")
        diff_tensor = torch.load(f"{out_dir}/steering_tensor.pt")
    else:
        print(f"...Creating steering tensor...")
        correct_prompts = correct["prompt"]
        incorrect_prompts = incorrect["prompt"]
        correct_avg_tensor = batched_get_averages(model, 
                                                correct_prompts,
                                                tokens=[-1],
                                                batch_size=args.batch_size)

        incorrect_avg_tensor = batched_get_averages(model,
                                                    incorrect_prompts,
                                                    tokens=[-1],
                                                    batch_size=args.batch_size)
        
        print(f"Correct avg tensor shape: {correct_avg_tensor.shape}")
        print(f"Incorrect avg tensor shape: {incorrect_avg_tensor.shape}")

        diff_tensor = correct_avg_tensor - incorrect_avg_tensor
        # diff_tensor = rearrange(diff_tensor, "l t d -> l 1 t d") # [n_layers, n_prompts, n_tokens, n_embd]

        print(f"Diff tensor shape after transform: {diff_tensor.shape}")

        torch.save(diff_tensor, f"{out_dir}/steering_tensor.pt")

    #==========================================================================================
    # Part 3: steered generations
    #==========================================================================================
    
    # def _evaluate(steering_ds, counts, args, out_dir, outfile):
    #     correct_steer = steering_ds.filter(lambda x : x["correct_steer"] == True)
    #     metric = f"{len(correct_steer)} / {len(steering_ds)} = {len(correct_steer) / len(steering_ds)}"
    #     print(metric)
    #     with open(f"{out_dir}/{outfile}", "w") as f:
    #         f.write(f"## Steering Results\n")
    #         f.write(metric)
    #         # write arguments of parser
    #         f.write(f"\n## Arguments\n")
    #         parser = vars(args)
    #         for k,v in parser.items():
    #             f.write(f"{k} : {v}\n")
    #         f.write("\nEval type distribution\n")
    #         f.write(str(counts))
            
    # print(f"...Applying patch to incorrect prompts...")
    # incorrect = datasets.Dataset.from_pandas(pd.DataFrame(incorrect))

    # steering_ds = steer(model, 
    #                     args,
    #                     incorrect,
    #                     diff_tensor,
    #                     args.layers_to_patch,
    #                     args.tokens_to_patch,
    #                     args.patch_mode,
    #                     args.batch_size)
    # steering_ds.save_to_disk(f"{out_dir}/steering_results_ds")
    
    # # save in multiple completions format
    # df = steering_ds.to_pandas()
    # df = df[["steered_generation","fim_type","correct_steer","generated_text", "hexsha"]]
    # df.to_csv(f"{out_dir}/steering_results.csv")
            
    # _evaluate(steering_ds, counts, args, out_dir, "eval_readme.md")
        
    # # ==========================================================================================
    # # PART 4: steering generation ood
    # # ==========================================================================================
    # if args.test_size > 0:
    #     print(f"...Applying patch to incorrect prompts...")

    #     incorrect_eval = datasets.Dataset.from_pandas(pd.DataFrame(incorrect_eval))
    #     counts = Counter(incorrect["fim_type"])
    #     print(counts)

    #     steering_ds = steer(model, 
    #                         args,
    #                         incorrect_eval,
    #                         diff_tensor,
    #                         args.layers_to_patch,
    #                         args.tokens_to_patch,
    #                         args.patch_mode,
    #                         args.batch_size)
    #     steering_ds.save_to_disk(f"{out_dir}/steering_results_ood")
    #     df = steering_ds.to_pandas()
    #     df = df[["steered_generation","fim_type","correct_steer","generated_text", "hexsha"]]
    #     df.to_csv(f"{out_dir}/steering_results_ood.csv")
        
    #     _evaluate(steering_ds, counts, args, out_dir, "eval_ood_readme.md")
    
    
if __name__ == "__main__":
    main()