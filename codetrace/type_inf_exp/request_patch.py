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
import json

def batched_get_averages(
    model: LanguageModel,
    prompts : List[str],
    tokens : Union[List[str],List[int]],
    batch_size=5,
    outfile = None
) -> torch.Tensor:
    """
    Get averages of tokens at all layers for all prompts
    
    NOTE:
    - if tokens is string, then first occurence of token is used
    - if tokens is int, then that token index is used
    """
    # batch prompts according to batch size
    prompt_batches = [prompts[i:i+batch_size] for i in range(0, len(prompts), batch_size)]
    hidden_states = []
    for i,batch in tqdm(enumerate(prompt_batches), desc="Batch", total=len(prompt_batches)):
        if tokens == []:
            hs = collect_hidden_states(model, batch)
        else:
            hs = collect_hidden_states_at_tokens(model, batch, tokens)
        hs_mean = hs.mean(dim=1) # batch size mean
        hidden_states.append(hs_mean)
        if outfile is not None:
            with open(outfile+".pkl", "wb") as f:
                pickle.dump(hidden_states, f)
            with open(outfile+".json", "w") as f:
                json.dump({"batch_size" : batch_size, "batch_idx" : i, "prompts" : prompt_batches}, f)
        
    # save tensor
    if tokens == []:
        # this means prompts are different token sizes, can't be stacked
        raise NotImplementedError("Prompts are different token sizes, need solution")
    
    hidden_states = torch.stack(hidden_states, dim=0)
    print(f"Hidden states shape before avg: {hidden_states.shape}")
    return hidden_states.mean(dim=0)


def batched_insert_patch(
    model : LanguageModel,
    prompts : Union[List[str],str],
    patch : torch.Tensor,
    layers_to_patch : List[int],
    tokens_to_patch : Union[List[str],List[int],str,int],
    patch_mode : str = "add",
    batch_size : int = 5
) -> List[TraceResult]:
    """
    batched insert patch
    """
    if tokens_to_patch == []:
        tokens_to_patch = list(range(patch.shape[1]))
    # batch prompts according to batch size
    prompt_batches = [prompts[i:i+batch_size] for i in range(0, len(prompts), batch_size)]
    results = []
    for i,batch in tqdm(enumerate(prompt_batches), desc="Insert Patch Batch", total=len(prompt_batches)):
        res : TraceResult = insert_patch(model, batch, patch, layers_to_patch, tokens_to_patch, patch_mode)
        results.append(res)
           
    return results


def batched_insert_patch_logit(
    model : LanguageModel,
    prompts : Union[List[str],str],
    patch : torch.Tensor,
    layers_to_patch : List[int],
    tokens_to_patch : Union[List[str],List[int],str,int],
    patch_mode : str = "add",
    batch_size : int = 5,
    outfile: str = None,
    solutions : Union[List[str],str, None] = None,
    custom_decoder : Union[torch.nn.Module, None] = None
) -> List[str]:
    """
    batched insert patch
    """
    def _percent_success(predictions_so_far, solutions):
        correct = 0
        for pred,sol in zip(predictions_so_far, solutions[:len(predictions_so_far)]):
            if pred.startswith(sol) or pred == sol:
                correct += 1
        return correct / len(solutions)
    
    if tokens_to_patch == []:
        tokens_to_patch = list(range(patch.shape[1]))
    # batch prompts according to batch size
    prompt_batches = [prompts[i:i+batch_size] for i in range(0, len(prompts), batch_size)]
    predictions =[]
    for i,batch in tqdm(enumerate(prompt_batches), desc="Insert Patch Batch", total=len(prompt_batches)):
        res : TraceResult = insert_patch(model, 
                                         batch, 
                                         patch, 
                                         layers_to_patch, 
                                         tokens_to_patch, 
                                         patch_mode, 
                                         collect_hidden_states=False,
                                         custom_decoder=custom_decoder)
        prompt_len = len(batch)
        logits : LogitResult = res.decode_logits(prompt_idx=list(range(prompt_len)), do_log_probs=False)

        for j in range(prompt_len):
            tok = logits[-1][j].tokens(model.tokenizer)
            tok = tok[0].strip()
            predictions.append(tok)
            
        if outfile is not None:
            with open(outfile, "w") as f:
                data = {"batch_size" : batch_size, "batch_idx" : i, "predictions" : predictions}
                if solutions is not None:
                    json.dump({"percent_success" : _percent_success(predictions, solutions), **data}, f, indent=4)
                else:
                    json.dump(data, f, indent=4)
           
    return predictions

def filter_prompts(
    dataset : datasets.Dataset,
    single_tokenize : PreTrainedTokenizer = None,
    dedup_prog_threshold : int = 3,
    dedup_type_threshold : int = 10
) -> datasets.Dataset:
    """
    Balance prompts s.t. there is a balanced distribution of labels.
    Do not use more than max_size prompts.
    Remove multi-token label prompts if tokenizer is passed.
    Deduplicate prompts by hexsha by some dedup_prog_threshold (max prompts for a program)
    """
    if not single_tokenize is None:
        dataset = dataset.map(lambda x: {"fim_type" : single_tokenize.decode(single_tokenize.encode(x["fim_type"])[0]), **x})
    # get count of labels
    labels = dataset["fim_type"]
    counter = Counter(labels)
    
    hexsha_count = {h:0 for h in dataset["hexsha"]}
    label_count = {label : 0 for label in labels}
    balanced_prompts = []
    for i,ex in enumerate(dataset):
        if label_count[ex["fim_type"]] >= dedup_type_threshold and hexsha_count[ex["hexsha"]] >= dedup_prog_threshold: 
            # if label and hexsha are already at threshold, break
            break
        elif label_count[ex["fim_type"]] >= dedup_type_threshold or hexsha_count[ex["hexsha"]] >= dedup_prog_threshold:
            # if hexsha is at threshold, continue
            continue
        
        balanced_prompts.append(ex)
        label_count[ex["fim_type"]] += 1
        hexsha_count[ex["hexsha"]] += 1

    df = pd.DataFrame(balanced_prompts)
    ds = datasets.Dataset.from_pandas(df)
    return ds