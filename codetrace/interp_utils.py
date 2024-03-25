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

def arg_to_list(x):
    return x if isinstance(x, list) else [x]

def arg_to_literal(x, n=24):
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
        if self.token_indices.squeeze().dim() > 1:
            raise ValueError(f"token_indices must be 1d")
        
        def _to_string(idx):
            return tokenizer.convert_tokens_to_string(tokenizer.tokenize(tokenizer.decode(idx)))
        
        return [_to_string(i.item()) for i in self.token_indices]
    
    def probs(self) -> np.ndarray:
        return self.probabilities.flatten().numpy()
    
class TraceResult:
    """
    Wrapper over hidden states and logits.
    
    Logits have shape [n_layer, n_prompt, n_tokens, n_vocab]
    Hidden states have shape [n_layer, n_prompt, n_tokens, n_embd]
    
    NOTE:
    - assumes logit comes unsoftmaxed
    - all tensors that get passed go through are detached and sent to cpu
    - input/output tensors are stacked.
    
    Except for dim -1 (n_vocab, n_embd), all other dims are variable
    
    A concern with this class is OOM errors. Temporary solution is to detach and move all tensors to cpu.
    can cause RAM errors if too many tensors are stored.
    """
    
    def __init__(self, 
                 logits : torch.Tensor, 
                 layer_idxs : Union[List[int],int],
                 hidden_states : torch.Tensor = None,
                 model_n_layer : int = 24,
                 custom_decoder : Union[None, torch.nn.Module] = None):
        self._logits = logits.detach().cpu()
        if hidden_states != None:
            hidden_states = hidden_states.detach().cpu()
        self._hidden_states = hidden_states
        self.n_layers = model_n_layer
        self._layer_idx = [arg_to_literal(i, n=model_n_layer) for i in arg_to_list(layer_idxs)]
        self.custom_decoder = custom_decoder
        
    def decode_logits(self, 
                    top_k : int = 1,
                    layers : Union[List[int],int] = -1,
                    prompt_idx : Union[List[int],int] = 0,
                    token_idx : Union[List[int],int] = -1,
                    do_log_probs : bool = False,
                    top_p : float = 1.0) -> LogitResult:
        """
        Decode logits to tokens, after scoring top_k
        NOTE: layer idxs are [0, n_layer)
        """
        layers, token_idx, prompt_idx = map(arg_to_list, [layers, token_idx, prompt_idx])
        layers = [self._layer_idx.index(arg_to_literal(i, n=self.n_layers)) for i in layers]
        token_idx = [arg_to_literal(i, self._logits.shape[2]) for i in token_idx]
        logits = self._logits.index_select(0, torch.tensor(layers)
                                           ).index_select(1, torch.tensor(prompt_idx)
                                                          ).index_select(2, torch.tensor(token_idx))
        if self.custom_decoder is not None:
            logits = self.custom_decoder(logits)
        
        logits = top_k_top_p_filtering(logits, top_k, top_p, do_log_probs)

        return LogitResult(logits.indices, logits.values, do_log_probs)


def collect_hidden_states(model : LanguageModel,
                          prompts : Union[List[str],str],
                          layers : List[int] = None) -> List[torch.Tensor]:
    """
    Collect hidden states for each prompt. 
    Optionally, collect hidden states at specific layers.
    """
    prompts = arg_to_list(prompts)
    if layers is None:
        layers = list(range(len(model.transformer.h)))

    with model.forward() as runner:
        with runner.invoke(prompts) as invoker:
            hidden_states = torch.stack([
                model.transformer.h[layer_idx].output[0]
                for layer_idx in layers
            ],dim=0).save()
            
    hidden_states = util.apply(hidden_states, lambda x: x.value, Proxy)
    return hidden_states


def collect_hidden_states_at_tokens(model : LanguageModel,
                                    prompts : Union[List[str],str],
                                    tokens : Union[List[int],int,List[str],str],
                                    layers : List[int] = None,
                                    debug = False) -> torch.Tensor:
    """
    Collect hidden states for each prompt at specific layers and tokens.
    
    NOTE:
    - if token is a string, select first occurence of that token in the prompt
    - if token is an int, it is an index. Select the index in the prompt
    """
    prompts, tokens = map(arg_to_list, [prompts, tokens])
    if layers is None:
        layers = list(range(len(model.transformer.h)))
    if isinstance(tokens[0], str):
        tokenized_idx = [model.tokenizer.encode(t)[0] for t in tokens]
        
    def decode(x : torch.Tensor) -> torch.Tensor:
        return model.lm_head(model.transformer.ln_f(x))

    with model.forward() as runner:
        with runner.invoke(prompts) as invoker:
            
            indices = invoker.input["input_ids"].numpy()
            
            if isinstance(tokens[0], str):
                target_idx = np.concatenate([np.where((i == t)) for t in tokenized_idx for i in indices], axis=0).reshape(indices.shape[0], -1)
            else:
                target_idx = np.array(tokens).reshape(indices.shape[0], -1)
                
            hidden_states = [
                    model.transformer.h[layer_idx].output[0]
                    for layer_idx in layers
                ]
            
            if debug:
                hidden_states = [decode(x) for x in hidden_states]
                
            hidden_states = torch.stack(hidden_states, dim=0).save()
            
    hidden_states = util.apply(hidden_states, lambda x: x.value.cpu(), Proxy)
    th = torch.tensor(target_idx)
    th = th.to(hidden_states.device)
    hidden_states = torch.stack([hidden_states[:,i,th[i],:] for i in range(len(prompts))], dim = 1)
    return hidden_states

def collect_attention_output(model : LanguageModel,
                             prompts : Union[List[str],str],
                             layers : List[int] = None) -> List[torch.Tensor]:
    """
    Collect attention output for each prompt at each layer.
    Note output will be padded to max sequence length
    """
    raise NotImplementedError("Not tested")

def insert_attn_patch(model : LanguageModel,
                 prompts : Union[List[str],str],
                 patch : torch.Tensor,
                 layers_to_patch : Union[List[int],int],
                 patch_mode : str = "add") -> TraceResult:
    """
    Insert attn patch at layers. Only supports full-token patch.
    Note that prompts will be padded to match the patch. If prompts are
    larger than patch, this will result in a mismatch.
    """
    raise NotImplementedError("Not tested")


def insert_patch(
    model : LanguageModel,
    prompts : Union[List[str],str],
    patch : torch.Tensor,
    layers_to_patch : Union[List[int],int],
    tokens_to_patch : Union[List[str],List[int],str,int],
    patch_mode : str = "add",
    collect_hidden_states : bool = True,
    custom_decoder : Union[None, torch.nn.Module] = None,
    rotation_matrix = None,
) -> TraceResult:
    """
    Insert patch at layers and tokens
    NOTE:
    - if token is a string, select first occurence of that token in the prompt
    - if token is an int, select the index of that token in the prompt
    """
    prompts, tokens_to_patch, layers_to_patch = map(arg_to_list, [prompts, tokens_to_patch, layers_to_patch])
    if patch.shape[0] != len(model.transformer.h):
        assert patch.shape[0] == len(layers_to_patch), f"Patch shape 0 {patch.shape} != len(layers_to_patch) {len(layers_to_patch)}"
    if patch.shape[2] != len(tokens_to_patch):
        assert patch.shape[2] == len(tokens_to_patch), f"Patch shape 2 {patch.shape} != len(tokens_to_patch) {len(tokens_to_patch)}"
    if isinstance(tokens_to_patch[0], str):
        tokenized_to_patch = [model.tokenizer.encode(t)[0] for t in tokens_to_patch]
    if patch_mode not in ["sub", "add", "subst"]:
        raise NotImplementedError(f"Patch mode {patch_mode} not implemented")
    
    # SPECIAL CASE: if token to patch is "*" patch everything
    if tokens_to_patch[0] == "*":
        print("[Apply to ALL]")
        apply_to_all = True
    else:
        apply_to_all = False
    
    def decode(x : torch.Tensor) -> torch.Tensor:
        if custom_decoder is not None:
            return x
        else:
            return model.lm_head(model.transformer.ln_f(x))
    
    with model.forward() as runner:
        with runner.invoke(prompts) as invoker:
            
            indices = invoker.input["input_ids"].numpy()

            if not apply_to_all:
                if isinstance(tokens_to_patch[0], str):
                    # for each prompt find the index of token_idx
                    target_idx = np.concatenate([np.where((i  == t)) for i in indices for t in tokenized_to_patch], axis=0).reshape((len(prompts),len(tokens_to_patch)))
                else:
                    target_idx = [tokens_to_patch]*len(prompts)
                
            # apply patch to hidden states at target_idx for each prompt
            for layer in range(len(model.transformer.h)):
                if layer in layers_to_patch: # don't touch
                    # grab patch
                    clean_patch = patch[layer]
                    # apply patch
                    def apply_patch(x : torch.Tensor, clean_patch) -> torch.Tensor:
                        for i in range(len(prompts)):
                            # if one unique patch for each prompt, grab the correct one
                            if clean_patch.shape[0] == len(prompts) and len(prompts)>1:
                                clean_patch = clean_patch.index_select(0, torch.tensor([i]))

                            if patch_mode == "subst":
                                if apply_to_all:
                                    x[[i],list(range(x.shape[1])),:] = clean_patch
                                else:
                                    x[[i],target_idx[i],:] = clean_patch
                            elif patch_mode == "add":
                                if apply_to_all:
                                    x[[i],list(range(x.shape[1])),:] += clean_patch
                                else:
                                    x[[i],target_idx[i],:] += clean_patch
                            elif patch_mode == "sub":
                                if apply_to_all:
                                    x[[i],list(range(x.shape[1])),:] -= clean_patch
                                else:
                                    x[[i],target_idx[i],:] -= clean_patch
                    
                    if rotation_matrix == None:
                        apply_patch(model.transformer.h[layer].output[0], clean_patch)
                    else:
                        batch_size = len(prompts)
                        out = rotation_matrix(model.transformer.h[layer].output[0][:,-1,:], clean_patch.to("cuda"))
                        hs = model.transformer.h[layer].output[0]
                        hs[:,-1,:] = out
                        model.transformer.h[layer].output = (hs, model.transformer.h[layer].output[1])
            
            if collect_hidden_states:
                collect_range = list(range(len(model.transformer.h)))
            else:
                collect_range = [-1]
                   
            hidden_states = torch.stack([
                model.transformer.h[layer_idx].output[0]
                for layer_idx in collect_range
            ],dim=0).save()
        
            logits = decode(hidden_states).save()
            
    hidden_states = util.apply(hidden_states, lambda x: x.value, Proxy)
    logits = util.apply(logits, lambda x: x.value, Proxy)
    
    return TraceResult(logits, collect_range, hidden_states, len(model.transformer.h), custom_decoder=custom_decoder)
    
            
def logit_lens(model : LanguageModel,
               prompts : Union[List[str],str],
               layers : Union[List[int],int] = None,
               apply_norm : bool = True,
               store_hidden_states : bool = False) -> TraceResult:
    """
    Apply logit lens to prompts
    """
    prompts = arg_to_list(prompts)
    if layers is None:
        layers = list(range(len(model.transformer.h)))
    else:
        layers = arg_to_list(layers, len(model.transformer.h))
        
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
        return TraceResult(logits, layers, hidden_states, len(model.transformer.h))
    else: 
        return TraceResult(logits, layers, model_n_layer=len(model.transformer.h))
    
    
def patch_clean_to_corrupt(model : LanguageModel,
                        clean_prompt : Union[List[str],str], 
                        corrupted_prompt : Union[List[str],str],
                        layers_to_patch : Union[List[int],int],
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
    return TraceResult(logits, -1, model_n_layer=len(model.transformer.h))
    

def custom_lens(model : LanguageModel,
                decoder : torch.nn.Module,
                prompts : Union[List[str],str],
                layer : Union[int,List[int]],
                activations : torch.Tensor = None,
                k : int = 1,) -> List[str]:
    """
    Apply custom lens to model activations for prompt at (layer, token)
    Note: for original decoder, load transformer version of model and copy
    """
    layer = arg_to_list(layer)
    prompts = arg_to_list(prompts)
    if activations is None:
        activations = collect_hidden_states(model,
                                            prompts,
                                            layers=layer)
    activations = activations.detach().cpu()
    # apply decoder
    logits = decoder(activations)

    # softmax
    logits = logits.softmax(dim=-1)
    topk= logits.topk(k, dim=-1)
    # output shape is [layer, prompt, tokens, vocab]
    
    return topk, activations, logits