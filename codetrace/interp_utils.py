"""
Interp utils
"""
import sys
import os
from codetrace.utils import *
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


class LogitResult:
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
        """
        TODO: torch like getitem eg. [:, 0, :, :]
        """
        return LogitResult(self.token_indices[idx,], self.probabilities[idx,], self.is_log)
    
    def tokens(self, tokenizer : transformers.PreTrainedTokenizer) -> List[str]:
        """
        Decode tokens to strings
        """
        # NOTE: this is only done once token_indices is a 1D tensor
        return [tokenizer.decode(i.item()) for i in self.token_indices]
    
    def probs(self) -> np.ndarray:
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
        if hidden_states != None:
            hidden_states = hidden_states.detach().cpu()
        self._hidden_states = hidden_states
        self._layer_idx = [arg_to_literal(i, NLAYER) for i in arg_to_list(layer_idxs)]
        
    def decode_logits(self, 
                    top_k : int = 1,
                    layers : List[int] | int = -1,
                    prompt_idx : List[int] | int = 0,
                    token_idx : List[int] | int = -1,
                    do_log_probs : bool = False) -> LogitResult:
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
        
        return LogitResult(logits.indices, logits.values, do_log_probs)


def collect_hidden_states(model : LanguageModel,
                          prompts : List[str] | str,
                          layers : List[int] = None,
                          target_module : str = "output") -> List[torch.Tensor]:
    """
    Collect hidden states for each prompt. 
    Optionally, collect hidden states at specific layers and tokens.
    """
    assert target_module in ["output", "attn"], f"target_module {target_module} not implemented"
    prompts = arg_to_list(prompts)
    if layers is None:
        layers = list(range(len(model.transformer.h)))

    with model.forward() as runner:
        with runner.invoke(prompts) as invoker:
            if target_module == "output":
                hidden_states = torch.stack([
                    model.transformer.h[layer_idx].output[0]
                    for layer_idx in layers
                ],dim=0).save()
            elif target_module == "attn":
                hidden_states = torch.stack([
                    model.transformer.h[layer_idx].attn.c_proj.output
                    for layer_idx in layers
                ],dim=0).save()
            
    hidden_states = util.apply(hidden_states, lambda x: x.value, Proxy)
    return hidden_states

def collect_hidden_states_at_tokens(model : LanguageModel,
                                    prompts : List[str] | str,
                                    token_idx : List[int] | int | List[str] | str,
                                    layers : List[int] = None,
                                    target_module : str = "output") -> torch.Tensor:
    """
    Collect hidden states for each prompt. 
    Optionally, collect hidden states at specific layers and tokens.
    
    NOTE: selects first occurence of token_idx in each prompt
    
    You can target following modules:
    - output (of attn+mlp)
    - attn (attention weights)
    """
    assert target_module in ["output", "attn"], f"target_module {target_module} not implemented"
    prompts, token_idx = map(arg_to_list, [prompts, token_idx])
    if layers is None:
        layers = list(range(len(model.transformer.h)))
    if isinstance(token_idx[0], str):
        token_idx = [model.tokenizer.encode(t)[0] for t in token_idx]
        
    def decode(x : torch.Tensor) -> torch.Tensor:
        return model.lm_head(model.transformer.ln_f(x))

    with model.forward() as runner:
        with runner.invoke(prompts) as invoker:
            
            indices = invoker.input["input_ids"].numpy()
            # for each prompt find the index of token_idx
            target_idx = np.concatenate([np.where((i  == t)) for t in token_idx for i in indices], axis=0).reshape(indices.shape[0], -1)
            
            if target_module == "output":
                hidden_states = torch.stack([
                    model.transformer.h[layer_idx].output[0]
                    for layer_idx in layers
                ],dim=0).save()
            elif target_module == "attn":
                hidden_states = torch.stack([
                    model.transformer.h[layer_idx].attn.c_proj.output
                    for layer_idx in layers
                ],dim=0).save()
            
    hidden_states = util.apply(hidden_states, lambda x: x.value.cpu(), Proxy)
    
    th = torch.tensor(target_idx)
    th = th.to(hidden_states.device)
    hidden_states = torch.stack([hidden_states[:,i,th[i],:] for i in range(len(prompts))], dim = 1)

    return hidden_states


def insert_patch(model : LanguageModel,
                 prompts : List[str] | str,
                 patch : torch.Tensor,
                 layers_to_patch : List[int],
                 tokens_to_patch : List[str] | List[int] | str | int,
                 module_to_patch : str = "output",
                 patch_mode : str = "add") -> TraceResult:
    """
    Insert patch at layers and tokens
    """
    assert module_to_patch in ["output", "attn"], f"module_to_patch {module_to_patch} not implemented"
    prompts, tokens_to_patch = arg_to_list(prompts), arg_to_list(tokens_to_patch)
    if patch.shape[0] != len(model.transformer.h):
        assert patch.shape[0] == len(layers_to_patch), f"Patch shape {patch.shape[0]} != len(layers_to_patch) {len(layers_to_patch)}"
    if patch.shape[2] != len(tokens_to_patch):
        assert patch.shape[2] == len(tokens_to_patch), f"Patch shape {patch.shape[2]} != len(tokens_to_patch) {len(tokens_to_patch)}"
    if isinstance(tokens_to_patch[0], str):
        tokens_to_patch = [model.tokenizer.encode(t)[0] for t in tokens_to_patch]
    if patch_mode not in ["sub", "add", "subst"]:
        raise NotImplementedError(f"Patch mode {patch_mode} not implemented")
    
    def decode(x : torch.Tensor) -> torch.Tensor:
        return model.lm_head(model.transformer.ln_f(x))
    
    with model.forward() as runner:
        with runner.invoke(prompts) as invoker:
            
            indices = invoker.input["input_ids"].numpy()
    
            # for each prompt find the index of token_idx
            target_idx = np.concatenate([np.where((i  == t)) for t in tokens_to_patch for i in indices], axis=0).reshape(indices.shape[0], -1)
            
            # apply patch to hidden states at target_idx for each prompt
            for layer in range(len(model.transformer.h)):
                if layer in layers_to_patch:
                    # grab patch
                    clean_patch = patch[layer]
                    # apply patch
                    def apply_patch(x : torch.Tensor) -> torch.Tensor:
                        for i in range(len(prompts)):
                            if patch_mode == "subst":
                                x[[i],target_idx[i],:] = clean_patch
                            elif patch_mode == "add":
                                x[[i],target_idx[i],:] += clean_patch
                            elif patch_mode == "sub":
                                x[[i],target_idx[i],:] -= clean_patch
                                
                    if module_to_patch == "output":
                        apply_patch(model.transformer.h[layer].output[0])
                    elif module_to_patch == "attn":
                        apply_patch(model.transformer.h[layer].attn.c_proj.output)
                            
            hidden_states = torch.stack([
                model.transformer.h[layer_idx].output[0]
                for layer_idx in range(len(model.transformer.h))
            ],dim=0).save()
            
            logits = decode(hidden_states).save()
            
    hidden_states = util.apply(hidden_states, lambda x: x.value, Proxy)
    logits = util.apply(logits, lambda x: x.value, Proxy)
    
    return TraceResult(logits, list(range(len(model.transformer.h))), hidden_states)
    
        
            
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
    if set(layers_to_patch).intersection(set(list(range(len(model.transformer.h))))) != set(layers_to_patch):
        raise ValueError(f"layers_to_patch must be in range [0, {len(model.transformer.h)}), received:\n{layers_to_patch}")
    if len(clean_prompt) != len(corrupted_prompt):
        raise ValueError(f"clean_prompt and corrupted_prompt must have same length")
    
    # Enter nnsight tracing context
    with model.forward() as runner:

        # Clean run
        with runner.invoke(clean_prompt) as invoker:
            
            # save all hidden states
            clean_hs = [
                model.transformer.h[layer_idx].output[0].save()
                for layer_idx in range(len(model.transformer.h))
            ]
        
        # Patch onto corrupted prompt
        logits = None
        with runner.invoke(corrupted_prompt) as invoker:
            
            for layer in layers_to_patch:
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
    