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

class TraceResult:
    """
    Wrapper over full component hidden states and logits
    """
    
    def __init__(self, logits : torch.Tensor, hidden_states : List[torch.Tensor] = None):
        self.logits = logits #[n_layer, n_prompt, n_token, n_vocab]
        self.hidden_states = hidden_states #[n_layer, n_prompt, n_token, n_embd]
        self._indexes = None
        self._values = None
        self._top_k = None
        
    def score_top(self, top_k : int = 10):
        """
        Decode logits to tokens
        """
        torch_topk = self.logits.topk(top_k)
        self._top_k = top_k
        self._indexes = torch_topk.indices
        self._values = torch_topk.values
        return self
    
    def _select(self, tensor, layers : List[int], token_idx : int, prompt_idx : int):
        """
        Select indexes
        """
        return tensor[layers, prompt_idx, token_idx, :]
        
        
    def get_tokens(self, 
                   model : LanguageModel,
                   layers : List[int],
                   token_idx : int = -1,
                   prompt_idx : int = 0) -> List[str]:
        """
        Decode logits to tokens
        """
        assert self._indexes is not None, "Run score() first"
        assert self._indexes.shape[0] == model.config.n_layer, "Expected logits from all model layers"
        assert self._top_k == 1, "Not implemented top_k>1"
        
        selected = self._indexes[layers, prompt_idx, token_idx] # returns a tensor of shape (layers, top_k)
        layer_tokens = []
        for i in range(len(layers)):
            layer_tokens.append(model.tokenizer.decode(selected[i]))
        return layer_tokens
    
    def get_indexes(self, 
                    layers : List[int], 
                    token_idx : int = -1,
                    prompt_idx : int =0) -> torch.Tensor:
        """
        Return indexes
        """
        return self._select(self._indexes, layers, token_idx, prompt_idx)
    
    def get_probabilities(self, 
                          layers : List[int], 
                          token_idx : int = -1, 
                          prompt_idx : int = 0, 
                          do_log_probs : bool = True) -> torch.Tensor:
        """
        Return values
        """
        values = self._values
        if do_log_probs:
            values = values.log()
        return self._select(values, layers, token_idx, prompt_idx)
        


def collect_hidden_states(model : LanguageModel,
                          prompts : List[str]) -> List[torch.Tensor]:
    """
    Collect hidden states for each prompt. By design does all layers
    """
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
               prompts : List[str],
               apply_norm : bool = True,
               store_hidden_states : bool = False) -> TraceResult:
    """
    Apply logit lens to prompts
    """
    def decode(x : torch.Tensor) -> torch.Tensor:
        if apply_norm:
            x = model.transformer.ln_f(x)
        return model.lm_head(x)
    
    layers = list(range(len(model.transformer.h)))
    with model.forward() as runner:
        with runner.invoke(prompts) as invoker:
            hidden_states = torch.stack([
                model.transformer.h[layer_idx].output[0]
                for layer_idx in layers
            ],dim=0).save()
            logits = decode(hidden_states).save()

            
    hidden_states = util.apply(hidden_states, lambda x: x.value, Proxy)
    logits = util.apply(logits, lambda x: x.value, Proxy)
    
    logits = logits.softmax(dim=-1)

    if store_hidden_states:
        return TraceResult(logits, hidden_states)
    else: 
        return TraceResult(logits)
    
    
def patch_clean_to_corrupt(model : LanguageModel,
                     clean_prompt : str, 
                     corrupted_prompt : str,
                     clean_index : int,
                     corrupted_index : int,
                     layers_to_patch : list[int],
                     apply_norm : bool = True,
                     store_hidden_states : bool = True) -> TraceResult:
    """
    Patch from clean prompt to corrupted prompt
    clean_idx -> corrupted_idx at target layers. Patches from same layers in clean prompt.
    
    Returns patched hidden states from corrupted run
    """
    def decode(x : torch.Tensor) -> torch.Tensor:
        if apply_norm:
            x = model.transformer.ln_f(x)
        return model.lm_head(x)
    
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
        hidden_states = []
        logits = []
        with runner.invoke(corrupted_prompt) as invoker:
            
            for layer in range(len(model.transformer.h)):
                if layer in layers_to_patch:
                    # grab patch
                    clean_patch = clean_hs[layer].t[clean_index]
                    # apply patch
                    model.transformer.h[layer].output[0].t[corrupted_index] = clean_patch

                # save patched hidden states
                hs = model.transformer.h[layer].output[0]
                hidden_states.append(hs.save())
                logits.append(decode(hs).save())
            
    hidden_states = util.apply(hidden_states, lambda x: x.value, Proxy)
    logits = util.apply(logits, lambda x: x.value, Proxy)
    logits = torch.stack(logits, dim=0)
    hidden_states = torch.stack(hidden_states, dim=0)
    
    logits = logits.softmax(dim=-1)
    
    return TraceResult(logits, hidden_states)