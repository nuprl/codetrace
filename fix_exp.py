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
from tqdm import tqdm
import pandas as pd

from scipy.stats import truncnorm

def sample_truncated_normal(high, low, mean, std_dev, size=100):
    # Set truncation limits for the first standard deviation
    a, b = (low - mean) / std_dev, (high - mean) / std_dev

    # Generate random numbers from a truncated normal distribution
    samples = truncnorm.rvs(a, b, loc=mean, scale=std_dev, size=size)
    samples = [int(s) for s in samples]
    return samples
    
def beam_mask_search(llm, dataset, logger):
    """
    Problem: Greedy alg is suboptimal. How to find optimal? -> SPRES + beam search. 
    Branch at different intervals by changing the random seed (SPRES). then 
    use a beam search to keep only top candidates. This finds the optimal mask?
    """
    np.random.seed(42)
    n_head = llm.config.n_head
    n_layer = llm.config.n_layer
    loc = n_layer // 2 # mean
    scale = n_layer // 2 # std
    samples = []
    causal_mask = np.zeros((n_layer, n_head))

    for i,ex in tqdm(enumerate(dataset)):
        i+=1
        prompt = ex["prompt"]
        correct_tok_idx = llm.tokenizer.encode(ex["fim_sol"])[0]
        incorrect_tok_idx = llm.tokenizer.encode(ex["generated_text"])[0]     

        tries = 0
        k = 150
        seen = set()
        while tries < k:
            tries += 1
            # sample a layer idx with gaussian sampling
            if len(samples) == 0:
                samples = list(sample_truncated_normal(n_layer, 0, n_layer // 2, 8, 1000))
                layer_idx = samples.pop()
            else:
                layer_idx = samples.pop()
            head_idx = np.random.randint(n_head)
            if (layer_idx, head_idx) in seen:
                continue
            seen.add((layer_idx, head_idx))
        
            causal_mask[layer_idx][head_idx] = 1
            mask = torch.tensor(causal_mask).float()
        
            correct_idx_prob, incorrect_idx_prob, max_prob = apply_mask(llm, 
                                                                        mask, 
                                                                        [prompt], 
                                                                        [correct_tok_idx], 
                                                                        [incorrect_tok_idx], 
                                                                        True)

            correct_idx_prob = util.apply(correct_idx_prob, lambda x: x.value.flatten().item(), Proxy)
            incorrect_idx_prob = util.apply(incorrect_idx_prob, lambda x: x.value.flatten().item(), Proxy)
            max_prob = util.apply(max_prob, lambda x: x.value.values.flatten().item(), Proxy)
                        
            if correct_idx_prob > before_correct_idx_prob and incorrect_idx_prob < before_incorrect_idx_prob:
                if logger:
                    wandb.log({"correct_idx_prob": correct_idx_prob, 
                                "incorrect_idx_prob": incorrect_idx_prob, 
                                "max_prob": max_prob, 
                                "layer_idx": layer_idx, 
                                "head_idx": head_idx,
                                "step": i})
                break
            else:
                causal_mask[layer_idx][head_idx] = 0
                
        if tries >= k:
            print("Tries exceeded 25")
            
        if i % 10 == 0:
            # save mask
            mask = torch.tensor(causal_mask).float()
            with open(f"masks/search/mask_{i}.pt", "wb") as f:
                torch.save(mask, f)
        
    with open("masks/search/mask.pt", "wb") as f:
        mask = torch.tensor(causal_mask).float()
        torch.save(mask, f)
    

def greedy_mask_search(llm, dataset, logger):
    """
    For each example:
    - sample a layer idx (make middle layers more likely with gaussian sampling)
    - sample a head idx
    - mask the head (layer, head)
    - if head makes correct token more likely, and/or incorrect token less likely, keep mask
        - otherwise, discard mask
        
    2/3 passes of fitting:
    1. correct_idx_prob > before_correct_idx_prob and incorrect_idx_prob < before_incorrect_idx_prob
    2. correct_idx_prob > incorrect_idx_prob
    3. correct_idx_prob > max_prob
    
    """
    np.random.seed(42)
    n_head = llm.config.n_head
    n_layer = llm.config.n_layer
    loc = n_layer // 2 # mean
    scale = n_layer // 2 # std
    samples = []
    zero_mask = torch.zeros((n_layer, n_head)).float()
    causal_mask = np.zeros((n_layer, n_head))
    new_ds = []
    seen = set()
    
    for i,ex in tqdm(enumerate(dataset)):
        i+=1
        prompt = ex["prompt"]
        correct_tok_idx = llm.tokenizer.encode(ex["fim_sol"])[0]
        incorrect_tok_idx = llm.tokenizer.encode(ex["generated_text"])[0]     
        
        before_correct_idx_prob, before_incorrect_idx_prob, before_max_prob = apply_mask(llm, 
                                                                                        zero_mask, 
                                                                                        [prompt], 
                                                                                        [correct_tok_idx], 
                                                                                        [incorrect_tok_idx], 
                                                                                        True)
        before_correct_idx_prob = util.apply(before_correct_idx_prob, lambda x: x.value.flatten().item(), Proxy)
        before_incorrect_idx_prob = util.apply(before_incorrect_idx_prob, lambda x: x.value.flatten().item(), Proxy)
        before_max_prob = util.apply(before_max_prob, lambda x: x.value, Proxy)
        before_max_prob_idx = before_max_prob.indices.item()
        before_max_prob_p = before_max_prob.values.item()
        # if within 4 decimal places, the max prob is the same as the incorrect token prob, skip
        if before_max_prob_idx != incorrect_tok_idx:
            print(f"{i}. Predicted token {llm.tokenizer.decode(before_max_prob_idx)} should be same as incorrect {ex['generated_text']}.")
            continue
        new_ds.append({**ex, **{"correct_idx_prob": before_correct_idx_prob, 
                            "incorrect_idx_prob": before_incorrect_idx_prob}})
        # causal_mask = np.zeros((n_layer, n_head))

        tries = 0
        k = 150
        while tries < k:
            tries += 1
            # sample a layer idx with gaussian sampling
            if len(samples) == 0:
                samples = list(sample_truncated_normal(n_layer, 0, n_layer // 2, 8, 1000))
                layer_idx = samples.pop()
            else:
                layer_idx = samples.pop()
            head_idx = np.random.randint(n_head)
            if (layer_idx, head_idx) in seen:
                continue
            seen.add((layer_idx, head_idx))
        
            causal_mask[layer_idx][head_idx] = 1
            mask = torch.tensor(causal_mask).float()
        
            correct_idx_prob, incorrect_idx_prob, max_prob = apply_mask(llm, 
                                                                        mask, 
                                                                        [prompt], 
                                                                        [correct_tok_idx], 
                                                                        [incorrect_tok_idx], 
                                                                        True)

            correct_idx_prob = util.apply(correct_idx_prob, lambda x: x.value.flatten().item(), Proxy)
            incorrect_idx_prob = util.apply(incorrect_idx_prob, lambda x: x.value.flatten().item(), Proxy)
            max_prob = util.apply(max_prob, lambda x: x.value.values.flatten().item(), Proxy)
                        
            if correct_idx_prob > before_correct_idx_prob and incorrect_idx_prob < before_incorrect_idx_prob:
                if logger:
                    wandb.log({"correct_idx_prob": correct_idx_prob, 
                                "incorrect_idx_prob": incorrect_idx_prob, 
                                "max_prob": max_prob, 
                                "layer_idx": layer_idx, 
                                "head_idx": head_idx,
                                "step": i})
                break
            else:
                causal_mask[layer_idx][head_idx] = 0
                
        if tries >= k:
            print("Tries exceeded 25")
            
        if i % 10 == 0:
            # save mask
            mask = torch.tensor(causal_mask).float()
            with open(f"masks/search/mask_{i}.pt", "wb") as f:
                torch.save(mask, f)
        
    with open("masks/search/mask.pt", "wb") as f:
        mask = torch.tensor(causal_mask).float()
        torch.save(mask, f)
    new_df = pd.DataFrame(new_ds)
    # save to csv
    new_df.to_csv("new_ds.csv")
    return new_df
        

def main(device, wandb_bool):
    if wandb_bool:
        wandb.init(project='interp_search', name="v0")
        
    dataset = datasets.load_dataset("franlucc/ts_bench_starcoder1b_funcfim_incorrect_uniq_v1", split="train")
    dataset.shuffle()
    
    starcoderbase_1b = "/home/arjun/models/starcoderbase-1b/"
    llm = LanguageModel(starcoderbase_1b, device_map=device)
    
    causal_mask = torch.zeros((llm.config.n_layer, llm.config.n_head)).float()
    
    greedy_mask_search(llm, dataset, wandb_bool)
    
    if wandb_bool:
        wandb.finish()
        
if __name__ == "__main__":
    device = sys.argv[1]
    device = f"cuda:{device}"
    wandb_bool = True
    main(device, wandb_bool)