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
from train import apply_mask

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
mask = torch.load("1d_causal_mask_epoch_1.pt")
# shuffle randomly mask
# mask_new = shuffle_tensor_along_dimension(mask, 0)
# while mask_new.equal(mask):
#     mask_new = shuffle_tensor_along_dimension(mask, 0)

print(mask.shape)
dataset = dataset.filter(lambda x: len(x["prompt"]) < 8000)
starcoderbase_1b = "/home/arjun/models/starcoderbase-1b/"
llm = LanguageModel(starcoderbase_1b, device_map="auto")

batch_size = 2
prompts = [d["prompt"] for d in dataset]
correct_idxs = [llm.tokenizer.encode(d["fim_sol"])[0] for d in dataset]
incorrect_idxs = [llm.tokenizer.encode(d["generated_text"])[0] for d in dataset]
train_data = list(zip(prompts, correct_idxs, incorrect_idxs))
# shuffle train data
random.shuffle(train_data)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=1)
    
results=[]
# traverse the dataset in 
for data in tqdm(train_loader):
    prompts, correct_idxs, incorrect_idxs = data

    out = apply_mask(llm, mask, prompts, correct_idxs, incorrect_idxs, debug=True)

    out = util.apply(out, lambda x: x.value, Proxy)
    max_probs = out[1]
    solutions = [llm.tokenizer.decode(idx) for idx in correct_idxs]
    generated = [llm.tokenizer.decode(idx) for idx in incorrect_idxs]
    predictions = [llm.tokenizer.decode(idx) for idx in max_probs]
    
    for solution, prediction, generated in zip(solutions, predictions, generated):
        results.append({"post_patch_prediction": prediction, "fim_sol": solution, "generated": generated})
        print("Prediction:", prediction, "Solution:", solution, "Generated:", generated)

succ_count = 0
for res in results:
    if res["post_patch_prediction"] == res["fim_sol"]:
        succ_count += 1

print("Success rate:", succ_count / len(results), "(", succ_count, "/", len(results), ")")
# print(mask_new)