from codetrace.type_inf_exp.request_patch import *
from codetrace.utils import *
from einops import rearrange
from argparse import ArgumentParser, Namespace

# if a json file was passed, parse args from json
if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    with open(sys.argv[1], "r") as f:
        args = json.load(f)
    args = Namespace(**args)
else:
    parser = ArgumentParser()
    parser.add_argument("--outdir_idx", type=int)
    parser.add_argument("--dataset", type=str, default="franlucc/starcoderbase-1b-completions_typeinf_analysis")
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
    # give options: block_out, attn_out
    parser.add_argument("--module_to_patch", type=str, default="block_out", choices=["block_out", "attn_out"])
    args = parser.parse_args()

exp_dir = "/home/franlucc/projects/codetrace/codetrace/type_inf_exp"
ds = datasets.load_dataset(args.dataset, split="train")
# filter some weird tree-sitter uncaught types: {}, this
# remove too large prompts for OOM
ds = ds.filter(lambda x : x["solution"] not in ["this", "{}"] and len(x["prompt"]) < 8000)

model = LanguageModel(args.model, device_map="cuda")

if not os.path.exists(f"{exp_dir}/exp_data/v{args.outdir_idx}"):
    os.makedirs(f"{exp_dir}/exp_data/v{args.outdir_idx}")
    
out_dir = f"{exp_dir}/exp_data/v{args.outdir_idx}"
# ==========================================================================================
# PART 1: filter
# ==========================================================================================
    
def _pretty_print(ds) -> str:
    df = pd.DataFrame(ds)
    s = ""
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        s += str(df["solution"].value_counts())
        s += "\n"
        s += str(df["hexsha"].value_counts())
        s += "\n"
        s += str(len(df))
    return s
    
# load if exists
if os.path.exists(f"{out_dir}/correct_prompts.csv"):
    correct = pd.read_csv(f"{out_dir}/correct_prompts.csv", encoding='utf-8')
else:
    correct = ds.filter(lambda x : x["correctness"] == "correct")
    correct = filter_prompts(correct, 
                         single_tokenize=model.tokenizer, 
                         dedup_prog_threshold=args.correct_prog_threshold, 
                         dedup_type_threshold=args.correct_type_threshold)
    
if os.path.exists(f"{out_dir}/incorrect_prompts.csv"):
    incorrect = pd.read_csv(f"{out_dir}/incorrect_prompts.csv", encoding='utf-8')
else:
    incorrect = ds.filter(lambda x : x["correctness"] == "incorrect")
    incorrect = filter_prompts(incorrect,
                                single_tokenize=model.tokenizer,
                                dedup_prog_threshold=args.incorrect_prog_threshold,
                                dedup_type_threshold=args.incorrect_type_threshold)
    
# save
correct.to_csv(f"{out_dir}/correct_prompts.csv", encoding='utf-8')
incorrect.to_csv(f"{out_dir}/incorrect_prompts.csv", encoding='utf-8')

sinc = _pretty_print(incorrect)
scorr = _pretty_print(correct)

with open(f"{out_dir}/data_readme.md", "w") as f:
    f.write(f"## Correct\n")
    f.write(scorr)
    f.write(f"## Incorrect\n")
    f.write(sinc)
    
print(sinc)
print(scorr)

# # ==========================================================================================
# # PART 2: averages
# # ==========================================================================================
print(f"...Getting averages for correct and incorrect prompts...")

df_correct = pd.read_csv(f"{out_dir}/correct_prompts.csv", encoding='utf-8')
df_incorrect = pd.read_csv(f"{out_dir}/incorrect_prompts.csv", encoding='utf-8')

# if exists, load
if os.path.exists(f"{out_dir}/{args.module_to_patch}_correct_avg_tensor.pt"):
    correct_avg_tensor = torch.load(f"{out_dir}/{args.module_to_patch}_correct_avg_tensor.pt")
else:
    correct_avg_tensor = batched_get_averages(model, df_correct['prompt'].tolist(), 
                                              args.tokens_to_patch, 
                                                batch_size=args.batch_size)
    # save tensor
    torch.save(correct_avg_tensor, f"{out_dir}/{args.module_to_patch}_correct_avg_tensor.pt")
    
if os.path.exists(f"{out_dir}/{args.module_to_patch}_incorrect_avg_tensor.pt"):
    incorrect_avg_tensor = torch.load(f"{out_dir}/{args.module_to_patch}_incorrect_avg_tensor.pt")
else:
    incorrect_avg_tensor = batched_get_averages(model, df_incorrect['prompt'].tolist(), 
                                                args.tokens_to_patch, 
                                                batch_size=args.batch_size)
    # save tensor
    torch.save(incorrect_avg_tensor, f"{out_dir}/{args.module_to_patch}_incorrect_avg_tensor.pt")
    
diff_tensor = correct_avg_tensor - incorrect_avg_tensor

print(f"Diff tensor shape before transform: {diff_tensor.shape}")
diff_tensor = rearrange(diff_tensor, "l t d -> l 1 t d") # [n_layers, n_prompts, n_tokens, n_embd]
print(f"Diff tensor shape after transform: {diff_tensor.shape}")

torch.save(diff_tensor, f"{out_dir}/{args.module_to_patch}_diff_tensor.pt")

#==========================================================================================
# PART 3: apply diff tensor to incorrect prompts, record top logit
#==========================================================================================
print(f"...Applying patch to incorrect prompts...")

df_incorrect = pd.read_csv(f"{out_dir}/incorrect_prompts.csv", encoding='utf-8')
args.n_eval = min(args.n_eval, len(df_incorrect))
# cap it at some size
df_incorrect = df_incorrect.sample(args.n_eval, random_state=2)


diff_tensor = torch.load(f"{out_dir}/{args.module_to_patch}_diff_tensor.pt")

batch_size = args.batch_size
prompts = df_incorrect['prompt'].tolist()
out = batched_insert_patch(model, 
                   prompts, 
                   diff_tensor, 
                   args.layers_to_patch,
                    args.tokens_to_patch,
                   patch_mode = args.patch_mode,
                   batch_size=batch_size)

batched_prompts, batched_labels, batched_old_generated, batched_ids = [], [], [], []
# batch df_incorrect
for i in range(0, len(prompts), batch_size):
    batched_prompts.append(prompts[i:i+batch_size])
    batched_labels.append(df_incorrect['solution'].tolist()[i:i+batch_size])
    batched_old_generated.append(df_incorrect['generated'].tolist()[i:i+batch_size])
    batched_ids.append(df_incorrect['id'].tolist()[i:i+batch_size])

steering_res = []
for i,trace_res in tqdm(enumerate(out), desc="Logits"):
    prompt_len = trace_res._logits.shape[1]
    logits : LogitResult = trace_res.decode_logits(prompt_idx=list(range(prompt_len)), do_log_probs=True)

    for j in list(range(prompt_len)):
        tok = logits[-1][j][-1].tokens(model.tokenizer)
        assert len(tok) == 1, tok
        tok = tok[0]
        if tok == batched_labels[i][j]:
            success = True
        else:
            success = False
        steering_res.append({"prompt" : batched_prompts[i][j], 
                             "label" : batched_labels[i][j],
                             "post_steer_toptok" : tok, 
                             "success" : success,
                             "generated" : batched_old_generated[i][j],
                            "id" : batched_ids[i][j]
                             })
    
with open(f"{out_dir}/{args.module_to_patch}_steering_res_{args.n_eval}.json", "w") as f:
    json.dump(steering_res, f, indent=4)
    
# ==========================================================================================
# # PART 4: plot steering results
# ==========================================================================================

with open(f"{out_dir}/{args.module_to_patch}_steering_res_{args.n_eval}.json", "r") as f:
    steering_res = json.load(f)
    
steering_res = pd.DataFrame(steering_res)
# count success
num_success = steering_res['success'].value_counts()
print(num_success)

with open(f"{out_dir}/{args.module_to_patch}_readme_{args.n_eval}.md", "w") as f:
    f.write(f"## Steering Results\n")
    f.write(num_success.to_string())
    # write arguments of parser
    f.write(f"\n## Arguments\n")
    parser = vars(args)
    for k,v in parser.items():
        f.write(f"{k} : {v}\n")