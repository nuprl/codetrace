"""
Interp utils
"""
import sys
import os
import torch
from tqdm import tqdm
from nnsight import LanguageModel, util
from nnsight.tracing.Proxy import Proxy
from typing import List, Union, Callable
import transformers
from codetrace.utils import top_k_top_p_filtering
import numpy as np

def arg_to_list(x):
    return x if isinstance(x, list) else [x]

def arg_to_literal(x, n):
    return x if x >= 0 else n + x

class LogitResult:
    """
    Wrapper over logits and token indices.
    Tensors in shape: [n_layer, n_prompt, n_tokens, topk]
    """
    def __init__(
        self, 
        token_indices : torch.Tensor, 
        probabilities : torch.Tensor,
        is_log : bool = False
    ):
        self.token_indices = token_indices
        self.probabilities = probabilities
        self.is_log = is_log
        
    def __getitem__(self, idx):
        return LogitResult(self.token_indices[idx], self.probabilities[idx], self.is_log)
    
    def tokens(self, tokenizer : transformers.PreTrainedTokenizer) -> List[str]:
        """
        Decode tokens to strings
        """
        if self.token_indices.squeeze().dim() > 1:
            raise ValueError(f"token_indices must be 1d")

        return tokenizer.decode(self.token_indices.flatten().numpy())


class TraceResult:
    """
    Wrapper over hidden states and logits.
    
    Arguments:
        logits : torch.Tensor
            logits from model, have shape [n_layer, n_prompt, n_tokens, n_vocab]
        layer_idxs : Union[List[int],int]
            from which model layers the logits were collected
        model_n_layer : int
            number of layers in model
        hidden_states : torch.Tensor
            hidden states from model, have shape [n_layer, n_prompt, n_tokens, n_embd]
        custom_decoder : Union[None, torch.nn.Module]
            custom decoder to apply to logits
    
    NOTE:
    - assumes logit input is unsoftmaxed
    - all tensors that get passed go through are detached and sent to cpu
    - input/output tensors are stacked
    
    A concern with this class is OOM errors. Temporary solution is to detach and move all tensors to cpu.
    can cause RAM errors if too many tensors are stored.
    """
    
    def __init__(
        self, 
        logits : torch.Tensor, 
        layer_idxs : Union[List[int],int],
        model_n_layer : int,
        hidden_states : torch.Tensor = None,
        custom_decoder : Union[None, torch.nn.Module] = None
    ):
        self._logits = logits.detach().cpu()
        if hidden_states != None:
            hidden_states = hidden_states.detach().cpu()
        self._hidden_states = hidden_states
        del(hidden_states)
        del(logits)
        self.n_layers = model_n_layer
        # if layer idxs are negative indexing, convert to positive index
        self._layer_idx = [arg_to_literal(i, n=model_n_layer) for i in arg_to_list(layer_idxs)]
        self.custom_decoder = custom_decoder
        
    def decode_logits(
        self, 
        top_k : int = 1,
        layers : Union[List[int],int] = -1,
        prompt_idx : Union[List[int],int] = 0,
        token_idx : Union[List[int],int] = -1,
        do_log_probs : bool = False,
        top_p : float = 1.0
    ) -> LogitResult:
        """
        Decode logits to tokens, after scoring top_k.
        NOTE: layers will index the literal layer index, not the index in the list. This means
        -1 corresponds to the last layer model_n_layer - 1.
        """
        layers, token_idx, prompt_idx = map(arg_to_list, [layers, token_idx, prompt_idx])
        # following will throw error if index is not present:
        layers = [self._layer_idx.index(arg_to_literal(i, n=self.n_layers)) for i in layers]
        token_idx = [arg_to_literal(i, self._logits.shape[2]) for i in token_idx]
        
        # select logits while maintaining len(shape) = 4
        logits = self._logits[layers][:,prompt_idx][:,:,token_idx]
        
        if self.custom_decoder is not None:
            logits = self.custom_decoder(logits)
        
        logits = top_k_top_p_filtering(logits, top_k, top_p, do_log_probs)
        return LogitResult(logits.indices, logits.values, do_log_probs)


def collect_hidden_states(
    model : LanguageModel,
    prompts : Union[List[str],str],
    layers : List[int] = None
) -> List[torch.Tensor]:
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
                model.transformer.h[layer_idx].output[0] # output is a tuple of (hidden_states, present)
                for layer_idx in layers
            ],dim=0).save()
            
    hidden_states = util.apply(hidden_states, lambda x: x.value, Proxy)
    return hidden_states


def collect_hidden_states_at_tokens(
    model : LanguageModel,
    prompts : Union[List[str],str],
    tokens : Union[List[int],int,List[str],str, Callable],
    layers : List[int] = None,
) -> torch.Tensor:
    """
    Collect hidden states for each prompt at specific tokens.
    Optionally, collect hidden states at specific layers.
    
    NOTE:
    tokens can be interpreted in two ways
    - str: select the index of this token in the tokenized prompt. 
        Assumes token appears exactly once in prompt, will throw error if token not found OR if multiple occurences.
    - int: select this index in the tokenized prompt
    """
    prompts, tokens = map(arg_to_list, [prompts, tokens])
    if layers is None:
        layers = list(range(len(model.transformer.h)))
    if isinstance(tokens[0], str):
        tokenized_idx = [model.tokenizer.encode(t)[0] for t in tokens]

    with model.forward() as runner:
        with runner.invoke(prompts) as invoker:
            
            indices = invoker.input["input_ids"].numpy()

            if isinstance(tokens[0], str):
                # for each prompt find the index of token_idx
                target_idx = np.array([np.where((i  == t)) for i in indices for t in tokenized_idx]).reshape((len(prompts),-1))
            elif callable(tokens[0]):
                tokens = tokens[0]
                target_idx = np.array([tokens(i, tokenizer=model.tokenizer) for i in indices]).reshape((len(prompts),-1)).astype(np.int64)
            else:
                target_idx = np.array(tokens*len(prompts)).reshape(len(prompts), -1)
            
            hidden_states = [
                    model.transformer.h[layer_idx].output[0]
                    for layer_idx in layers
                ]
                
            hidden_states = torch.stack(hidden_states, dim=0).save() 
            
    hidden_states = util.apply(hidden_states, lambda x: x.value, Proxy)
    th = torch.tensor(target_idx).to(hidden_states.device)
    hidden_states = torch.stack([hidden_states[:,i,th[i],:] for i in range(len(prompts))], dim = 1)
    return hidden_states

def insert_patch(
    model : LanguageModel,
    prompts : Union[List[str],str],
    patch : torch.Tensor,
    layers_to_patch : Union[List[int],int],
    tokens_to_patch : Union[List[str],List[int],str,int, Callable],
    patch_mode : str = "add",
    collect_hidden_states : bool = True,
    custom_decoder : Union[None, torch.nn.Module] = None,
) -> TraceResult:
    """
    Insert patch at layers and tokens
    NOTE:
    tokens can be interpreted in two ways
    - str: select the index of this token in the tokenized prompt. 
        Assumes token appears exactly once in prompt, will throw error if token not found OR if multiple occurences.
    - int: select this index in the tokenized prompt
    
    patch should have shape [model_n_layer, num_prompts, num_tokens_to_patch, n_embd]
    """
    prompts, tokens_to_patch, layers_to_patch = map(arg_to_list, [prompts, tokens_to_patch, layers_to_patch])
    if not callable(tokens_to_patch[0]) and patch.shape != (len(model.transformer.h), len(prompts), len(tokens_to_patch), model.config.hidden_size):
        raise ValueError(f"Patch shape {patch.shape} is incorrect, requires {len(model.transformer.h), len(prompts), len(tokens_to_patch), model.config.hidden_size}")
    if patch_mode not in ["sub", "add", "subst"]:
        raise NotImplementedError(f"Patch mode {patch_mode} not implemented")
    
    if isinstance(tokens_to_patch[0], str):
        tokenized_to_patch = [model.tokenizer.encode(t)[0] for t in tokens_to_patch]
    
    def decode(x : torch.Tensor) -> torch.Tensor:
        if custom_decoder is not None:
            return x
        else:
            return model.lm_head(model.transformer.ln_f(x))
    
    with model.forward() as runner:
        with runner.invoke(prompts) as invoker:
            
            indices = invoker.input["input_ids"].numpy()
            
            if isinstance(tokens_to_patch[0], str):
                # for each prompt find the index of token_idx
                target_idx = np.array([np.where((i  == t)) for i in indices for t in tokenized_to_patch]).reshape((len(prompts),-1))
            elif callable(tokens_to_patch[0]):
                tokens_to_patch = tokens_to_patch[0]
                target_idx = np.array([tokens_to_patch(i, tokenizer=model.tokenizer) for i in indices]).reshape((len(prompts),-1))
            else:
                target_idx = np.array(tokens_to_patch*len(prompts)).reshape(len(prompts), -1)
            
            # apply patch to hidden states at target_idx for each prompt
            for layer in range(len(model.transformer.h)):
                if layer in layers_to_patch:
                    
                    def apply_patch(x : torch.Tensor) -> torch.Tensor:
                        # grab layer patch
                        layer_patch = patch[layer]
                        
                        for i in range(len(prompts)):
                            # grab prompt patch
                            prompt_patch = layer_patch[[i]]
                            if patch_mode == "subst":
                                x[[i],target_idx[i],:] = prompt_patch
                            elif patch_mode == "add":
                                x[[i],target_idx[i],:] += prompt_patch
                            elif patch_mode == "sub":
                                x[[i],target_idx[i],:] -= prompt_patch
                    
                    apply_patch(model.transformer.h[layer].output[0])
            
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
    
    return TraceResult(logits, collect_range, len(model.transformer.h), hidden_states=hidden_states, custom_decoder=custom_decoder)
    
            
def logit_lens(
    model : LanguageModel,
    prompts : Union[List[str],str],
    layers : Union[List[int],int] = None,
    apply_norm : bool = True,
    store_hidden_states : bool = False
) -> TraceResult:
    """
    Apply logit lens to prompts
    """
    prompts = arg_to_list(prompts)
    if layers is None:
        layers = list(range(len(model.transformer.h)))
    else:
        layers = arg_to_list(layers)
        layers = [arg_to_literal(x, len(model.transformer.h)) for x in layers]
        
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

    logits = util.apply(logits, lambda x: x.value, Proxy)
    
    if store_hidden_states:
        hidden_states = util.apply(hidden_states, lambda x: x.value, Proxy)
        return TraceResult(logits, layers, len(model.transformer.h), hidden_states=hidden_states)
    else: 
        return TraceResult(logits, layers, model_n_layer=len(model.transformer.h))
    

def custom_lens(
    model : LanguageModel,
    decoder : torch.nn.Module,
    prompts : Union[List[str],str],
    layers : Union[int,List[int]],
    activations : torch.Tensor = None,
    k : int = 1
) -> List[str]:
    """
    Apply custom lens to model activations for prompt at (layer, token)
    Note: for original decoder, load transformer version of model and copy
    """
    layers, prompts = map(arg_to_list, [layers, prompts])

    if activations is None:
        activations = collect_hidden_states(model,prompts,layers=layers)
    else:
        activations = activations[layers]
    
    # apply decoder
    activations = activations.detach().cpu()
    logits = decoder(activations)

    # softmax
    logits = logits.softmax(dim=-1)
    topk= logits.topk(k, dim=-1)
    # output shape is [layer, prompt, tokens, vocab]
    return topk, activations, logits