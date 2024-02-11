from nnsight import LanguageModel, util
from nnsight.tracing.Proxy import Proxy
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import sys
from src.utils import *
import einops
from typing import List
import random
import numpy as np
import datasets
import wandb
from datasets import Dataset
from train import apply_mask, z_score_normalize

def shuffle_tensor(tensor):
    # Generate a random permutation of indices
    indices = torch.randperm(tensor.numel())

    # Reshape the tensor to a 1D tensor, shuffle using the indices, and reshape back
    shuffled_tensor = tensor.view(-1)[indices].view(tensor.size())

    return shuffled_tensor


def shuffle_tensor_along_dimension(tensor, dim):
    # Get the size of the specified dimension
    dim_size = tensor.size(dim)

    # Generate a random permutation of indices along the specified dimension
    indices = torch.randperm(dim_size)

    # Use the shuffled indices to permute the specified dimension
    shuffled_tensor = tensor.index_select(dim, indices)

    return shuffled_tensor

os.environ["TOKENIZERS_PARALLELISM"] = "false"
dataset = datasets.load_dataset("franlucc/ts_bench_starcoder1b_funcfim_incorrect_uniq", split="train")


mask = torch.load("masks/success_maybe_attn_only/causal_mask_epoch_1.pt")
# shuffle randomly mask
# mask_new = shuffle_tensor_along_dimension(mask, 0)
# while mask_new.equal(mask):
#     mask_new = shuffle_tensor_along_dimension(mask, 0)

print(mask.shape)
dataset = dataset.filter(lambda x: len(x["prompt"]) < 8000)
starcoderbase_1b = "/home/arjun/models/starcoderbase-1b/"
llm = LanguageModel(starcoderbase_1b, device_map=f"cuda:{sys.argv[1]}")

batch_size = 2
prompts = [d["prompt"] for d in dataset]
correct_idxs = [llm.tokenizer.encode(d["fim_sol"])[0] for d in dataset]
incorrect_idxs = [llm.tokenizer.encode(d["generated_text"])[0] for d in dataset]
idxs = list(zip(correct_idxs, incorrect_idxs))
train_data = list(zip(prompts, idxs))

# shuffle train data
random.shuffle(train_data)


train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=2)



results=[]
mask_n = z_score_normalize(mask)
mask_n = (mask_n > 0.5).float()
# print(mask_n)
mask = torch.zeros((llm.config.n_layer, llm.config.n_head)).float()
# traverse the dataset in 
for data in train_loader:
    
    prompts, (correct_idxs, incorrect_idxs) = data
    
    
    # print("Prompts:", prompts)
    # print("Correct idxs:", correct_idxs)
    # print("Incorrect idxs:", incorrect_idxs)

    correct_idx_p, incorrect_idx_p, max_probs = apply_mask(llm, mask, prompts, correct_idxs, incorrect_idxs, debug=True)

    correct_idx_p = util.apply(correct_idx_p, lambda x: x.value, Proxy)
    incorrect_idx_p = util.apply(incorrect_idx_p, lambda x: x.value, Proxy)
    # print("Correct idxs:", correct_idx_p)
    # print("Incorrect idxs:", incorrect_idx_p)
    max_probs = util.apply(max_probs, lambda x: x.value, Proxy)

    solutions = [llm.tokenizer.decode(idx) for idx in correct_idxs]
    generated = [llm.tokenizer.decode(idx) for idx in incorrect_idxs]
    predictions = [llm.tokenizer.decode(idx) for idx in max_probs.indices]
    
    for solution, prediction, generated in zip(solutions, predictions, generated):
        results.append({"post_patch_prediction": prediction, "fim_sol": solution, "generated": generated})
        print("Prediction:", prediction, "Solution:", solution, "Generated:", generated)

succ_count = 0
unchanged = 0
for res in results:
    if res["post_patch_prediction"] == res["fim_sol"]:
        succ_count += 1
    if res["post_patch_prediction"] == res["generated"]:
        unchanged +=1

print("Success rate:", succ_count / len(results), "(", succ_count, "/", len(results), ")")
print("Unchanged rate:", unchanged / len(results), "(", unchanged, "/", len(results), ")")