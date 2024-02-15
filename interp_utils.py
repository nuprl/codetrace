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

def arg_to_list(x):
    return x if isinstance(x, list) else [x]

def to_literal_idx(x, n):
    return x if x >= 0 else n+x

class Logit:
    """
    tensors in shape [n_layer, n_prompt, n_tokens, topk]
    """
    def __init__(self, 
                 tokenizer : transformers.PreTrainedTokenizer,
                 token_indices : torch.Tensor, 
                 probabilities : torch.Tensor,
                 is_log : bool = False):
        self.tokenizer = tokenizer
        self.token_indices = token_indices
        self.probabilities = probabilities
        self.is_log = is_log
        
    def __getitem__(self, idx):
        return Logit(self.tokenizer, self.token_indices[idx,], self.probabilities[idx,], self.is_log)
    
    def tokens(self):
        """
        Decode tokens to strings
        """
        # NOTE: this is only done once token_indices is a 1D tensor
        return [self.tokenizer.decode(i.item()) for i in self.token_indices]
    
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
        self._layer_idx = arg_to_list(layer_idxs)
        
    def decode_logits(self, 
                    model : LanguageModel,
                    top_k : int = 1,
                    layers : List[int] | int = -1,
                    prompt_idx : List[int] | int = 0,
                    token_idx : List[int] | int = -1,
                    do_log_probs : bool = False) -> Logit:
        """
        Decode logits to tokens, after scoring top_k
        """
        layers, token_idx, prompt_idx = map(arg_to_list, [layers, token_idx, prompt_idx])
        # find idx of layers in self._layer_idx
        n = model.config.n_layer
        layers = list(map(lambda x : to_literal_idx(x,n), layers))
        token_idx = list(map(lambda x : to_literal_idx(x, self._logits.shape[2]), token_idx))
        
        logits = self._logits.index_select(0, torch.tensor(layers)).index_select(1, torch.tensor(prompt_idx)).index_select(2, torch.tensor(token_idx))
        print(logits.shape)
        
        if do_log_probs:
            logits = logits.softmax(dim=-1).log()
        else:
            logits = logits.softmax(dim=-1)
        logits = logits.topk(top_k, dim=-1)
        
        return Logit(model.tokenizer, logits.indices, logits.values, do_log_probs)


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
    
    if store_hidden_states:
        return TraceResult(logits, layers, hidden_states)
    else: 
        return TraceResult(logits, layers)
    
    
# def patch_clean_to_corrupt(model : LanguageModel,
#                      clean_prompt : str, 
#                      corrupted_prompt : str,
#                      clean_index : int,
#                      corrupted_index : int,
#                      layers_to_patch : list[int],
#                      apply_norm : bool = True,
#                      store_hidden_states : bool = False) -> TraceResult:
#     """
#     Patch from clean prompt to corrupted prompt
#     clean_idx -> corrupted_idx at target layers. Patches from same layers in clean prompt.
    
#     Returns patched hidden states from corrupted run
#     """
#     def decode(x : torch.Tensor) -> torch.Tensor:
#         if apply_norm:
#             x = model.transformer.ln_f(x)
#         return model.lm_head(x)
    
#     # Enter nnsight tracing context
#     with model.forward() as runner:

#         # Clean run
#         with runner.invoke(clean_prompt) as invoker:
            
#             clean_tokens = invoker.input["input_ids"][0]
            
#             # save all hidden states
#             clean_hs = [
#                 model.transformer.h[layer_idx].output[0].save()
#                 for layer_idx in range(len(model.transformer.h))
#             ]
        
#         # Patch onto corrupted prompt
#         hidden_states = []
#         logits = []
#         with runner.invoke(corrupted_prompt) as invoker:
            
#             for layer in range(len(model.transformer.h)):
#                 if layer in layers_to_patch:
#                     # grab patch
#                     clean_patch = clean_hs[layer].t[clean_index]
#                     # apply patch
#                     model.transformer.h[layer].output[0].t[corrupted_index] = clean_patch

#                 # save patched hidden states
#                 hs = model.transformer.h[layer].output[0]
#                 hidden_states.append(hs.save())
#                 logits.append(decode(hs).save())
            
#     hidden_states = util.apply(hidden_states, lambda x: x.value, Proxy)
#     logits = util.apply(logits, lambda x: x.value, Proxy)
#     hidden_states = torch.stack(hidden_states, dim=0)
    
#     logits = torch.stack(logits, dim=0)
#     logits = logits.softmax(dim=-1)
#     return TraceResult(logits, hidden_states)
    