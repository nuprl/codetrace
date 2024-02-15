"""
Interp utils
"""
import sys
import os
from utils import *
import glob
import datasets
import pandas as pd
import torch
from tqdm import tqdm
from nnsight import LanguageModel,util
from nnsight.tracing.Proxy import Proxy
from typing import List
import transformers
import copy

NLAYER = 24

def arg_to_list(x):
    return x if isinstance(x, list) else [x]

def arg_to_literal(x, n=NLAYER):
    return x if x >= 0 else n + x

class Logit:
    """
    tensors in shape [n_layer, n_prompt, n_tokens, topk]
    """
    def __init__(self, 
                 token_indices : torch.Tensor, 
                 probabilities : torch.Tensor,
                 is_log : bool = False):
        self.token_indices = token_indices
        self.probabilities = probabilities
        self.is_log = is_log
        
    def __getitem__(self, idx):
        return Logit(self.token_indices[idx,], self.probabilities[idx,], self.is_log)
    
    def tokens(self, tokenizer : transformers.PreTrainedTokenizer):
        """
        Decode tokens to strings
        """
        # NOTE: this is only done once token_indices is a 1D tensor
        return [tokenizer.decode(i.item()) for i in self.token_indices]
    
    def probs(self):
        return self.probabilities.flatten().numpy()
    
class TraceResult:
    """
    Wrapper over hidden states and logits.
    
    NOTE:
    - assumes logit comes unsoftmaxed
    - Logits have shape [n_layer, n_prompt, n_tokens, n_vocab]
    - Hidden states have shape [n_layer, n_prompt, n_tokens, n_embd]
    - all tensors that get passed go through detach and cpu
    Tensors are stacked.
    Except for dim -1, all other dims are variable
    
    A concern with this class is OOM errors. temporary solution is to detach and move all tensors to cpu
    """
    
    def __init__(self, 
                 logits : torch.Tensor, 
                 layer_idxs : List[int] | int,
                 hidden_states : torch.Tensor = None):
        self._logits = logits.detach().cpu()
        self._hidden_states = hidden_states.detach().cpu() if hidden_states else None
        self._layer_idx = [arg_to_literal(i, NLAYER) for i in arg_to_list(layer_idxs)]
        
    def decode_logits(self, 
                    top_k : int = 1,
                    layers : List[int] | int = -1,
                    prompt_idx : List[int] | int = 0,
                    token_idx : List[int] | int = -1,
                    do_log_probs : bool = False) -> Logit:
        """
        Decode logits to tokens, after scoring top_k
        NOTE: layer idxs are [0, n_layer)
        """
        layers, token_idx, prompt_idx = map(arg_to_list, [layers, token_idx, prompt_idx])
        layers = [self._layer_idx.index(arg_to_literal(i)) for i in layers]
        token_idx = [arg_to_literal(i, self._logits.shape[2]) for i in token_idx]
        logits = self._logits.index_select(0, torch.tensor(layers)
                                           ).index_select(1, torch.tensor(prompt_idx)
                                                          ).index_select(2, torch.tensor(token_idx))
        
        if do_log_probs:
            logits = logits.softmax(dim=-1).log()
        else:
            logits = logits.softmax(dim=-1)
        logits = logits.topk(top_k, dim=-1)
        
        return Logit(logits.indices, logits.values, do_log_probs)


def collect_hidden_states(model : LanguageModel,
                          prompts : List[str] | str ) -> List[torch.Tensor]:
    """
    Collect hidden states for each prompt. By design does all layers
    """
    prompts = arg_to_list(prompts)
    layers = list(range(len(model.transformer.h)))

    with model.forward() as runner:
        with runner.invoke(prompts) as invoker:
            hidden_states = torch.stack([
                model.transformer.h[layer_idx].output[0]
                for layer_idx in layers
            ],dim=0).save()
            
    hidden_states = util.apply(hidden_states, lambda x: x.value, Proxy)
    return hidden_states
        
            
def logit_lens(model : LanguageModel,
               prompts : List[str] | str,
               layers : List[int] | int = None,
               apply_norm : bool = True,
               store_hidden_states : bool = False) -> TraceResult:
    """
    Apply logit lens to prompts
    """
    prompts = arg_to_list(prompts)
    if layers is None:
        layers = list(range(len(model.transformer.h)))
    else:
        layers = arg_to_list(layers)
        
    def decode(x : torch.Tensor) -> torch.Tensor:
        if apply_norm:
            x = model.transformer.ln_f(x)
        return model.lm_head(x)
    
    with model.forward() as runner:
        with runner.invoke(prompts) as invoker:
            hidden_states = torch.stack([
                model.transformer.h[layer_idx].output[0]
                for layer_idx in layers
            ],dim=0).save()
            logits = decode(hidden_states).save()

    hidden_states = util.apply(hidden_states, lambda x: x.value, Proxy)
    logits = util.apply(logits, lambda x: x.value, Proxy)
    
    layers = [arg_to_literal(x, len(model.transformer.h)) for x in layers]
    if store_hidden_states:
        return TraceResult(logits, layers, hidden_states)
    else: 
        return TraceResult(logits, layers)
    
    
def patch_clean_to_corrupt(model : LanguageModel,
                        clean_prompt : List[str] | str, 
                        corrupted_prompt : List[str] | str,
                        layers_to_patch : List[int] | int,
                        clean_index : int = -1,
                        corrupted_index : int = -1,
                        apply_norm : bool = True) -> TraceResult:
    """
    Patch from a clean prompt to a corrupted prompt at target layers [cumulative]
        (layers_to_patch, clean_index) -> (layers_to_patch, corrupted_index)
        
    NOTE: Returns logits only from final layer
    """
    clean_prompt, corrupted_prompt, layers_to_patch = map(arg_to_list, [clean_prompt, corrupted_prompt, layers_to_patch])
    # Enter nnsight tracing context
    with model.forward() as runner:

        # Clean run
        with runner.invoke(clean_prompt) as invoker:
            
            clean_tokens = invoker.input["input_ids"][0]
            
            # save all hidden states
            clean_hs = [
                model.transformer.h[layer_idx].output[0].save()
                for layer_idx in range(len(model.transformer.h))
            ]
        
        # Patch onto corrupted prompt
        logits = None
        with runner.invoke(corrupted_prompt) as invoker:
            
            for layer in range(len(model.transformer.h)):
                if layer in layers_to_patch:
                    # grab patch
                    clean_patch = clean_hs[layer].t[clean_index]
                    # apply patch
                    model.transformer.h[layer].output[0].t[corrupted_index] = clean_patch

            # save final logits
            logits = model.lm_head.output.save()
                
    logits = util.apply(logits, lambda x: x.value, Proxy)
    # logit should be in shape [n_layer, n_prompt, n_tokens, n_vocab]
    logits = torch.stack([logits], dim=0)
    return TraceResult(logits, -1)
    