"""
Steering utils:
- batched patch requests
- prompt filtering
"""
from nnsight import LanguageModel
import torch
from tqdm import tqdm
from collections import Counter
import pickle
import json
import datasets
from typing import List, Union, Callable
from codetrace.interp_utils import (
    collect_hidden_states,
    insert_patch,
    TraceResult,
    LogitResult
)
import pandas as pd

def batched_get_averages(
    model: LanguageModel,
    prompts : List[str],
    target_fn: Callable,
    batch_size=5,
    outfile = None
) -> torch.Tensor:
    """
    Get averages of activations at all layers for prompts. Select activations according to mask
    produced by target_fn. Batches the prompts to
    avoid memory issues. If an outfile is passed, will cache the hidden states to the outfile.
    """
    # batch prompts according to batch size
    prompt_batches = [prompts[i:i+batch_size] for i in range(0, len(prompts), batch_size)]
    hidden_states = []
    for i,batch in tqdm(enumerate(prompt_batches), desc="Batch average", total=len(prompt_batches)):
        hs = collect_hidden_states(model, batch, target_fn).cpu()
        hs_mean = hs.mean(dim=1) # batch size mean
        hidden_states.append(hs_mean)
        if outfile is not None:
            with open(outfile+".pkl", "wb") as f:
                pickle.dump(hidden_states, f)
            with open(outfile+".json", "w") as f:
                json.dump({"batch_size" : batch_size, "batch_idx" : i, "prompts" : prompt_batches}, f)
        
    # save tensor
    hidden_states = torch.stack(hidden_states, dim=0)
    print(f"Hidden states shape before avg: {hidden_states.shape}")
    return hidden_states.mean(dim=0)

def _percent_success(predictions_so_far, solutions):
    correct = 0
    for pred,sol in zip(predictions_so_far, solutions[:len(predictions_so_far)]):
        if sol == pred:
            correct += 1
    return correct / len(solutions)
    
def batched_insert_patch_logit(
    model : LanguageModel,
    prompts : List[str],
    patch : torch.Tensor,
    layers_to_patch : List[int],
    target_fn : Callable,
    batch_size : int = 5,
    outfile: str = None,
    solutions : Union[List[str],str, None] = None,
) -> List[str]:
    """
    Inserts patch and collects resulting logits. Batches the prompts to avoid memory issues.
    If outfile and solutions are passed, will cache the predictions and accuracy to the outfile.
    """
    # batch prompts according to batch size
    prompt_batches = [prompts[i:i+batch_size] for i in range(0, len(prompts), batch_size)]
    predictions =[]
    for i,batch in tqdm(enumerate(prompt_batches), desc="Insert Patch Batch", total=len(prompt_batches)):
        # repeat patch in dim 1 to match batch len
        prompt_len = len(batch)
        res : TraceResult = insert_patch(model, 
                                         batch, 
                                         patch.repeat(1,prompt_len,1,1),
                                         layers_to_patch, 
                                         target_fn=target_fn,
                                         collect_hidden_states=False, # don't need hidden states, prevent oom
                                         )
        logits : LogitResult = res.decode_logits(prompt_idx=list(range(prompt_len)), layers=[-1], token_idx=[-1])

        for j in range(prompt_len):
            tok = logits[-1,j].tokens(model.tokenizer).strip()
            predictions.append(tok)
            
        if outfile is not None:
            with open(outfile, "w") as f:
                data = {"batch_size" : batch_size, "batch_idx" : i, "total_batches": len(prompt_batches), "predictions" : predictions}
                if solutions is not None:
                    curr_accuracy =  _percent_success(predictions, solutions)
                    if i == 0:
                        projected_accuracy = 0
                    else:
                        projected_accuracy = (len(prompt_batches) * curr_accuracy) / i
                    json.dump({"current_accuracy" : curr_accuracy, "projected_accuracy": projected_accuracy, **data}, f, indent=4)
                else:
                    json.dump(data, f, indent=4)
           
    return predictions