"""
We try to find a model edit at the representation level that
is able to fix broken type-inf predictions.
"""
from nnsight import LanguageModel
from transformers import PreTrainedTokenizer
from typing import List
import sys
import os
from codetrace.interp_utils import *
from codetrace.interp_vis import *
import torch
from tqdm import tqdm
from collections import Counter, defaultdict
import random
import pickle

def batched_get_averages(model: LanguageModel,
                 prompts : List[str],
                 tokens : List[str],
                 batch_size=5) -> torch.Tensor:
    """
    Get averages of tokens at all layers for all prompts
    
    NOTE: if tokens aren't unique in the prompts, the first occurence is used.
    """
    tokens = [model.tokenizer.encode(t)[0] for t in tokens]
    # batch prompts according to batch size
    prompt_batches = [prompts[i:i+batch_size] for i in range(0, len(prompts), batch_size)]
    hidden_states = []
    for batch in tqdm(prompt_batches, desc="Batch"):
        if tokens == []:
            hs = collect_hidden_states(model, batch)
        else:
            hs = collect_hidden_states_at_tokens(model, batch, tokens)
        hs_mean = hs.mean(dim=1) # batch size mean
        hidden_states.append(hs_mean)
        
    # save tensor
    hidden_states = torch.stack(hidden_states, dim=0)
    print(f"Hidden states shape before avg: {hidden_states.shape}")
    return hidden_states.mean(dim=0)


def batched_insert_patch(model : LanguageModel,
                    prompts : List[str] | str,
                    patch : torch.Tensor,
                    layers_to_patch : List[int],
                    tokens_to_patch : List[str] | List[int] | str | int,
                    patch_mode : str = "add",
                    batch_size : int = 5) -> List[TraceResult]:
    """
    batched insert patch
    """
    if tokens_to_patch == []:
        tokens_to_patch = list(range(patch.shape[1]))
    # batch prompts according to batch size
    prompt_batches = [prompts[i:i+batch_size] for i in range(0, len(prompts), batch_size)]
    results = []
    for batch in tqdm(prompt_batches, desc="Batch"):
        res : TraceResult = insert_patch(model, batch, patch, layers_to_patch, tokens_to_patch, patch_mode)
        results.append(res)
        
    return results
    

def filter_prompts(dataset : datasets.Dataset,
                   single_tokenize : PreTrainedTokenizer = None,
                   dedup_prog_threshold : int = 3,
                   dedup_type_threshold : int = 10) -> datasets.Dataset:
    """
    Balance prompts s.t. there is a balanced distribution of labels.
    Do not use more than max_size prompts.
    Remove multi-token label prompts if tokenizer is passed.
    Deduplicate prompts by hexsha by some dedup_prog_threshold (max prompts for a program)
    """
    if not single_tokenize is None:
        dataset = dataset.filter(lambda x : len(single_tokenize.encode(x["fim_type"])) == 1)
    
    # get count of labels
    labels = dataset["fim_type"]
    counter = Counter(labels)
    
    hexsha_count = {h:0 for h in dataset["hexsha"]}
    label_count = {label : 0 for label in labels}
    balanced_prompts = []
    for i,ex in enumerate(dataset):
        if label_count[ex["fim_type"]] >= dedup_type_threshold:
            continue
        if hexsha_count[ex["hexsha"]] >= dedup_prog_threshold: # some threshold
            continue
        balanced_prompts.append(ex)
        label_count[ex["fim_type"]] += 1
        hexsha_count[ex["hexsha"]] += 1

    df = pd.DataFrame(balanced_prompts)
    ds = datasets.Dataset.from_pandas(df)
    return ds