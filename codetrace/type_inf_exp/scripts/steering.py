from codetrace.type_inf_exp.request_patch import *
from codetrace.utils import *
from einops import rearrange
from argparse import ArgumentParser, Namespace
from collections import Counter

# def _balance(ds):
#     new_ds = []
#     # traverse ds in pairs
#     # if one pair of hexsha is different, delete the first
#     i = 0
#     while i < (len(ds) - 1):
#         if ds[i]["hexsha"] == ds[i+1]["hexsha"]:
#             new_ds.append(ds[i])
#             new_ds.append(ds[i+1])
#             i += 2
#         else:
#             i += 1
#     return datasets.Dataset.from_pandas(pd.DataFrame(new_ds))

def fit_test_split(dataset : datasets.Dataset, tokenizer, args):
    correct = dataset.filter(lambda x : x["correct"] == True or x["correct"] == "true")
    incorrect = dataset.filter(lambda x : x["correct"] == False or x["correct"] == "false")

    # filter for balance and single token labels
    correct = filter_prompts(correct, 
                             single_tokenize=tokenizer, 
                             dedup_prog_threshold=args.correct_prog_threshold, 
                             dedup_type_threshold=args.correct_type_threshold)
    incorrect = filter_prompts(incorrect,
                                single_tokenize=tokenizer,
                                dedup_prog_threshold=args.incorrect_prog_threshold,
                                dedup_type_threshold=args.incorrect_type_threshold)
    
    if args.test_size > 0:
        # set aside some incorrect prompts
        random.seed(42)
        hexshas = list(incorrect["hexsha"])
        hexshas = random.sample(hexshas, int(len(hexshas) * args.test_size))
        incorrect_eval = incorrect.filter(lambda x : x["hexsha"] in hexshas)
        incorrect = incorrect.filter(lambda x : x["hexsha"] not in hexshas)
        correct = correct.filter(lambda x : x["hexsha"] not in hexshas)
        return correct, incorrect, incorrect_eval
    else:
        return correct, incorrect, incorrect


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
    # if a json file was passed, parse args from json
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        with open(sys.argv[1], "r") as f:
            args = json.load(f)
        args = Namespace(**args)
    else:
        parser = ArgumentParser()
        parser.add_argument("--outdir", type=int)
        parser.add_argument("--dataset", type=str, default="franlucc/stenotype-type-inference-fim-evaluated")
        parser.add_argument("--model", type=str, default="/home/arjun/models/starcoderbase-1b")
        parser.add_argument("--correct_prog_threshold", type=int, default=100)
        parser.add_argument("--correct_type_threshold", type=int, default=100)
        parser.add_argument("--incorrect_prog_threshold", type=int, default=100)
        parser.add_argument("--incorrect_type_threshold", type=int, default=100)
        parser.add_argument("--batch_size", type=int, default=2)
        parser.add_argument("--patch_mode", type=str, default="add")
        parser.add_argument("--n_eval", type=int, default=15)
        parser.add_argument("--tokens_to_patch", type=str, nargs="+", default=[])
        parser.add_argument("--layers_to_patch", type=int, nargs="+", default=[])
        parser.add_argument("--test_size", type=float, default=0.2)
        args = parser.parse_args()

    exp_dir = "/home/franlucc/projects/codetrace/codetrace/type_inf_exp"
    ds = datasets.load_dataset(args.dataset, split="train")
    
    # filter out too large prompts for OOM
    ds = ds.filter(lambda x : len(x["fim_program"]) < 8000)

    model = LanguageModel(args.model, device_map="cuda")

    out_dir = f"{exp_dir}/exp_data/{args.outdir}"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # ==========================================================================================
    # PART 1: filter
    # ==========================================================================================
    # print("...Generating fit test split...")
    
    # correct, incorrect, incorrect_eval = fit_test_split(ds,model.tokenizer, args)

    # info_incorrect = _pretty_print(incorrect)
    # info_correct = _pretty_print(correct)
    # info_eval = _pretty_print(incorrect_eval)

    # info = f"Correct\n{info_correct}\n\nIncorrect\n{info_incorrect}\n\nIncorrect Eval\n{info_eval}\n"

    # with open(f"{out_dir}/data_readme.md", "w") as f:
    #     f.write(info)
        
    # print(info)

    # # # ==========================================================================================
    # # # PART 2: averages
    # # # ==========================================================================================
    # print(f"...Getting averages for correct and incorrect prompts...")

    # correct_prompts = [placeholder_to_std_fmt(ex["fim_program"], STARCODER_FIM) for ex in correct]
    # incorrect_prompts = [placeholder_to_std_fmt(ex["fim_program"], STARCODER_FIM) for ex in incorrect]
    # correct_avg_tensor = batched_get_averages(model, 
    #                                         correct_prompts,
    #                                         args.tokens_to_patch,
    #                                             batch_size=args.batch_size)

    # incorrect_avg_tensor = batched_get_averages(model,
    #                                             incorrect_prompts,
    #                                             args.tokens_to_patch,
    #                                             batch_size=args.batch_size)
                                            
    # diff_tensor = correct_avg_tensor - incorrect_avg_tensor
    # diff_tensor = rearrange(diff_tensor, "l t d -> l 1 t d") # [n_layers, n_prompts, n_tokens, n_embd]

    # print(f"Diff tensor shape after transform: {diff_tensor.shape}")

    # torch.save(diff_tensor, f"{out_dir}/steering_tensor.pt")

    # #==========================================================================================
    # # PART 3: steering
    # #==========================================================================================
    # print(f"...Applying patch to incorrect prompts...")

    # args.n_eval = min(args.n_eval, len(incorrect_eval))
    # incorrect_eval = datasets.Dataset.from_pandas(pd.DataFrame(incorrect_eval).sample(args.n_eval, random_state=42))
    # counts = Counter(incorrect_eval["fim_type"])
    # print(counts)

    # steering_ds = steer(model, args,
    #                       incorrect_eval,
    #                     diff_tensor,
    #                     args.layers_to_patch,
    #                     args.tokens_to_patch,
    #                     args.patch_mode,
    #                     args.batch_size)
    # steering_ds.save_to_disk(f"{out_dir}/steering_results_ds")
    # df = steering_ds.to_pandas()
    # df = df[["steered_generation","fim_type","correct_steer","generated_text"]]
    # df.to_csv(f"{out_dir}/steering_results.csv")
    
    # # ==========================================================================================
    # # # PART 4: evaluate steering effects
    # # ==========================================================================================

    # correct_steer = steering_ds.filter(lambda x : x["correct_steer"] == True)
    # metric = f"{len(correct_steer)} / {len(steering_ds)} = {len(correct_steer) / len(steering_ds)}"
    # print(metric)
    # with open(f"{out_dir}/eval_readme.md", "w") as f:
    #     f.write(f"## Steering Results\n")
    #     f.write(metric)
    #     # write arguments of parser
    #     f.write(f"\n## Arguments\n")
    #     parser = vars(args)
    #     for k,v in parser.items():
    #         f.write(f"{k} : {v}\n")
    #     f.write("\nEval type distribution\n")
    #     f.write(str(counts))
        
    # ==========================================================================================
    # # PART 5: evaluate ood
    # ==========================================================================================
    print(f"...Applying patch to OOD prompts...")
    diff_tensor = torch.load(f"{out_dir}/steering_tensor.pt")
    ood_ds = datasets.load_dataset("franlucc/stenotype-type-inference-fim-evaluated", split="train")
    
    def _filter_ood(x):
        incorrect = x["correct"] == False or x["correct"] == "false"
        single_tok = len(model.tokenizer.tokenize(x["fim_type"])) == 1
        generated_text = not x["generated_text"].startswith(x["fim_type"])
        return incorrect and single_tok and generated_text
    
    ood_ds = ood_ds.filter(lambda x : _filter_ood(x))
    assert len(ood_ds) > 0, "No ood examples found"
    
    # sample n_eval from ood
    args.n_eval = 100
    ood_ds = datasets.Dataset.from_pandas(pd.DataFrame(ood_ds).sample(args.n_eval, random_state=42))
    
    counts = Counter(ood_ds["fim_type"])
    print(counts)
    
    steering_ood_ds = steer(
                        model,
                        args,
                        ood_ds,
                        diff_tensor,
                        args.layers_to_patch,
                        args.tokens_to_patch,
                        args.patch_mode,
                        args.batch_size)
    
    steering_ood_ds.save_to_disk(f"{out_dir}/steering_ood_results_ds")
    df = steering_ood_ds.to_pandas()
    df = df[["steered_generation","fim_type","correct_steer","generated_text"]]
    df.to_csv(f"{out_dir}/steering_ood_results.csv")
    
    correct_steer_ood = steering_ood_ds.filter(lambda x : x["correct_steer"] == True)
    metric = f"{len(correct_steer_ood)} / {len(steering_ood_ds)} = {len(correct_steer_ood) / len(steering_ood_ds)}"
    print(metric)
    with open(f"{out_dir}/ood_eval_readme.md", "w") as f:
        f.write(f"## Steering Results\n")
        f.write(metric)
        # write arguments of parser
        f.write(f"\n## Arguments\n")
        parser = vars(args)
        for k,v in parser.items():
            f.write(f"{k} : {v}\n")
        f.write("\nEval type distribution\n")
        f.write(str(counts))
    
if __name__ == "__main__":
    main()