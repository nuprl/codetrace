from codetrace.type_inf_exp.request_patch import *
from codetrace.utils import *

exp_dir = "/home/franlucc/projects/codetrace/codetrace/type_inf_exp"
ds = datasets.load_dataset("franlucc/starcoderbase-1b-completions_typeinf_analysis", split="train")
model = LanguageModel("/home/arjun/models/starcoderbase-1b", device_map="cuda")

# ==========================================================================================
# PART 1: filter
# ==========================================================================================

# def _filter(ds, t, p) -> datasets.Dataset:
#     return filter_prompts(prompts, labels, hexshas,
#                         dedup_type_threshold=t,
#                         dedup_prog_threshold=p,
#                         single_tokenize=model.tokenizer)
    
# correct = ds.filter(lambda x : x["correctness"] == "correct")
# incorrect = ds.filter(lambda x : x["correctness"] == "incorrect")

# correct = filter_prompts(correct, 
#                          single_tokenize=model.tokenizer, 
#                          dedup_prog_threshold=1, 
#                          dedup_type_threshold=2)
# incorrect = filter_prompts(incorrect,
#                             single_tokenize=model.tokenizer,
#                             dedup_prog_threshold=4,
#                             dedup_type_threshold=6)

# # filter some weird tree-sitter uncaught types: {}, this
# incorrect = incorrect.filter(lambda x : x["solution"] not in ["this", "{}"])

# correct.to_csv(f"{exp_dir}/exp_data/correct_prompts.csv", encoding='utf-8')
# incorrect.to_csv(f"{exp_dir}/exp_data/incorrect_prompts.csv", encoding='utf-8')

# # ==========================================================================================
# # PART 2: averages
# # ==========================================================================================

# df_correct = pd.read_csv(f"{exp_dir}/exp_data/correct_prompts.csv", encoding='utf-8')
# df_incorrect = pd.read_csv(f"{exp_dir}/exp_data/incorrect_prompts.csv", encoding='utf-8')
# # compute averages
# correct_avg_tensor = get_averages(model, df_correct['prompt'].tolist(), STARCODER_FIM.to_list()[:-1])
# # save tensor
# torch.save(correct_avg_tensor, f"{exp_dir}/exp_data/correct_avg_tensor.pt")

# incorrect_avg_tensor = get_averages(model, df_incorrect['prompt'].tolist(), STARCODER_FIM.to_list()[:-1])
# # save tensor
# torch.save(incorrect_avg_tensor, f"{exp_dir}/exp_data/incorrect_avg_tensor.pt")
    
# diff_tensor = correct_avg_tensor - incorrect_avg_tensor
# torch.save(diff_tensor, f"{exp_dir}/exp_data/diff_tensor.pt")

# ==========================================================================================
# # PART 3: apply diff tensor to incorrect prompts, record top logit
# ==========================================================================================

df_incorrect = pd.read_csv(f"{exp_dir}/exp_data/incorrect_prompts.csv", encoding='utf-8')
# cap it at some size
df_incorrect = df_incorrect.sample(10, random_state=2)
# remove too large prompts
print(df_incorrect)
df_incorrect = df_incorrect[df_incorrect['prompt'].apply(lambda x : len(x) < 8000)]
print(df_incorrect)

diff_tensor = torch.load(f"{exp_dir}/exp_data/diff_tensor.pt")

# diff tensor only last tok id, without changing shape
diff_tensor = diff_tensor.index_select(1, torch.tensor([2]))


prompts = df_incorrect['prompt'].tolist()
batch_size = 2
out = batched_insert_patch(model, 
                   prompts, 
                   diff_tensor, 
                   [14],
                #    STARCODER_FIM.to_list()[1:-1],
                    STARCODER_FIM.token,
                   patch_mode = "add",
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
    
with open(f"{exp_dir}/exp_data/steering_res.json", "w") as f:
    json.dump(steering_res, f, indent=4)
    
# ==========================================================================================
# # PART 4: plot steering results
# ==========================================================================================

with open(f"{exp_dir}/exp_data/steering_res.json", "r") as f:
    steering_res = json.load(f)
    
steering_res = pd.DataFrame(steering_res)
# count success
print(steering_res['success'].value_counts())