"""
Steering utils:
- batched patch requests
- prompt filtering
"""
import einops
from nnsight import LanguageModel
import torch
from tqdm import tqdm
import pickle
import json
from typing import List, Union, Callable, Optional, TypeVar
from codetrace.interp_utils import (
    collect_hidden_states,
    insert_patch,
    apply_reduction,
    TraceResult,
    LogitResult,
    HiddenStateStack_1tok,
    HiddenStateStack,
    InputMaskTensor,
    InputTensor,
    HiddenState,
    HiddenState_1tok,
    MaskTensor
)

def batched_patch(
    model : LanguageModel,
    prompts : Union[List[str], torch.utils.data.DataLoader],
    patch : Union[HiddenStateStack,HiddenStateStack_1tok],
    layers_to_patch : List[int],
    target_fn : Callable[[InputTensor],InputMaskTensor],
    patch_fn: Callable[[HiddenState, MaskTensor, HiddenState],HiddenState],
    batch_size : int = 5,
    reduction: Optional[Union[str, Callable[[HiddenState,int],HiddenState_1tok]]] = None,
    collect_hidden_states: Optional[List[int]] = None
) -> List[Union[HiddenStateStack, HiddenStateStack_1tok]]:
    if isinstance(prompts, torch.utils.data.DataLoader):
        prompt_batches = prompts
    else:
        prompt_batches = [prompts[i:i+batch_size] for i in range(0, len(prompts), batch_size)]
    
    hs = []
    for i,batch in tqdm(enumerate(prompt_batches), desc="Insert Patch Batch", total=len(prompt_batches)):
        bsize = len(batch) if isinstance(batch, List) else batch.shape[1]

        res : TraceResult = insert_patch(
            model, 
            batch, 
            _resize_patch(patch, bsize), # repeat patch in dim 1 to match batch len
            layers_to_patch, 
            target_fn=target_fn,
            patch_fn=patch_fn,
            collect_hidden_states=collect_hidden_states,
        )
        activs = res._hidden_states
        if reduction:
            activs = apply_reduction(activs, reduction, dim=1)
        hs.append(activs)
    return hs

def batched_collect_activations(
    model: LanguageModel,
    prompts : Union[List[str], torch.utils.data.DataLoader],
    target_fn : Optional[Callable[[InputTensor],InputMaskTensor]] = None,
    batch_size:int=5,
    layers: Optional[List[int]] = None,
    reduction: Optional[Union[str, Callable[[HiddenState,int],HiddenState_1tok]]] = None
) -> List[Union[HiddenStateStack, HiddenStateStack_1tok]]:
    """
    Get activations at all layers for prompts. Select activations according to mask
    produced by target_fn. Batches the prompts to avoid memory issues.

    reduction: provide an einops reduction function for reducing the collected activation to
            save gpu memory.
    """
    # batch prompts according to batch size
    if isinstance(prompts, torch.utils.data.DataLoader):
        prompt_batches = prompts
    else:
        prompt_batches = [prompts[i:i+batch_size] for i in range(0, len(prompts), batch_size)]

    if not layers:
        layers = range(model.config.num_hidden_layers)
    hidden_states = []
    for i,batch in tqdm(enumerate(prompt_batches), desc="Batch collect", total=len(prompt_batches)):
        hs = collect_hidden_states(model, batch,layers, target_fn, reduction=reduction).cpu()
        hidden_states.append(hs)
        
    # save tensor
    return hidden_states

def batched_get_averages(
    model: LanguageModel,
    prompts : Union[List[str], torch.utils.data.DataLoader],
    target_fn : Optional[Callable[[InputTensor],InputMaskTensor]] = None,
    batch_size:int =5,
    average_fn: Callable = (lambda x: x.mean(dim=1, keepdim=True)),
    outfile: Optional[str] = None,
    layers: Optional[List[int]] = None,
    reduction: Optional[Union[str, Callable[[HiddenState,int],HiddenState_1tok]]] = None
) -> Union[HiddenStateStack, HiddenStateStack_1tok]:
    """
    Get averages of activations at all layers for prompts. Select activations according to mask
    produced by target_fn. Batches the prompts to
    avoid memory issues. If an outfile is passed, will cache the hidden states to the outfile.

    reduction: provide an einops reduction function for reducing the collected activation to
            save gpu memory.
    """
    # batch prompts according to batch size
    if isinstance(prompts, torch.utils.data.DataLoader):
        prompt_batches = prompts
    else:
        prompt_batches = [prompts[i:i+batch_size] for i in range(0, len(prompts), batch_size)]

    if not layers:
        layers = range(model.config.num_hidden_layers)
    hidden_states = []
    for i,batch in tqdm(enumerate(prompt_batches), desc="Batch average", total=len(prompt_batches)):
        hs = collect_hidden_states(model, batch,layers, target_fn, reduction=reduction).cpu()
        hs_mean = average_fn(hs) # batch size mean
        hidden_states.append(hs_mean)
        if outfile is not None:
            torch.save(torch.stack(hidden_states, dim=0), outfile+".pt")
            with open(outfile+".json", "w") as f:
                json.dump({"batch_size" : batch_size, "batch_idx" : i, "prompts" : list(prompt_batches)}, f)
        
    # save tensor
    hidden_states = torch.stack(hidden_states, dim=0)
    print(f"Hidden states shape before avg: {hidden_states.shape}")
    return hidden_states.mean(dim=0)

def batched_insert_patch_logit(
    model : LanguageModel,
    prompts : Union[List[str], torch.utils.data.DataLoader],
    patch : Union[HiddenStateStack,HiddenStateStack_1tok],
    layers_to_patch : List[int],
    target_fn : Callable[[InputTensor],InputMaskTensor],
    patch_fn: Callable[[HiddenState, MaskTensor, HiddenState],HiddenState],
    batch_size : int = 5,
    outfile: str = None,
    solutions : Union[List[str],str, None] = None,
) -> List[str]:
    """
    Inserts patch and collects resulting logits. Batches the prompts to avoid memory issues.
    If outfile and solutions are passed, will cache the predictions and accuracy to the outfile.
    """
    # batch prompts according to batch size
    if isinstance(prompts, torch.utils.data.DataLoader):
        prompt_batches = prompts
    else:
        prompt_batches = [prompts[i:i+batch_size] for i in range(0, len(prompts), batch_size)]
    predictions =[]
    for i,batch in tqdm(enumerate(prompt_batches), desc="Insert Patch Batch", total=len(prompt_batches)):
        bsize = len(batch) if isinstance(batch, List) else batch.shape[1]

        res : TraceResult = insert_patch(
            model, 
            batch, 
            _resize_patch(patch, bsize), # repeat patch in dim 1 to match batch len
            layers_to_patch, 
            target_fn=target_fn,
            patch_fn=patch_fn,
            collect_hidden_states=False, # don't need hidden states, prevent oom
        )
        logits : LogitResult = res.decode_logits(prompt_idx=list(range(bsize)), layers=[-1], token_idx=[-1])

        for j in range(bsize):
            tok = logits[-1,j].tokens(model.tokenizer).strip()
            predictions.append(tok)
        
        # log
        if solutions:
            print("current_accuracy:", _percent_success(predictions, solutions))
        if outfile is not None:
            with open(outfile, "w") as f:
                data = {"batch_size" : batch_size, 
                        "batch_idx" : i, 
                        "total_batches": len(prompt_batches), 
                        "predictions" : predictions}
                json.dump(data, f, indent=4)
           
    return predictions

def _percent_success(predictions_so_far, solutions):
    correct = 0
    for pred,sol in zip(predictions_so_far, solutions[:len(predictions_so_far)]):
        if sol == pred:
            correct += 1
    return correct / len(solutions)

def _resize_patch(patch: Union[HiddenStateStack, HiddenStateStack_1tok],batch_size:int)->HiddenStateStack:
    if patch.ndim == 4 and patch.shape[1] == batch_size:
        return patch
    return einops.repeat(patch, "l t d -> l p t d", p=batch_size)
