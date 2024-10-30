"""
Interp utils
"""
import torch
import nnsight
from nnsight import LanguageModel, util
from nnsight.tracing.Proxy import Proxy
from typing import List, Union, Callable
import transformers
from codetrace.utils import top_k_top_p_filtering, pos_indexing, masked_get, masked_fill, lm_decode
import numpy as np
import functools
import einops

class LogitResult:
    """
    Wrapper over logits and token indices.
    Tensors in shape: [n_layer, n_prompt, n_tokens, topk]
    """
    def __init__(
        self, 
        token_indices : torch.Tensor, 
        probabilities : torch.Tensor,
    ):
        self.token_indices = token_indices
        self.probabilities = probabilities
        
    def __getitem__(self, idx):
        return LogitResult(self.token_indices[idx], self.probabilities[idx])
    
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
        layer_idxs : List[int]
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
        layer_idxs : List[int],
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
        self._layer_idx = [pos_indexing(i, n=model_n_layer) for i in layer_idxs]
        self.custom_decoder = custom_decoder
        
    def decode_logits(
        self, 
        prompt_idx : List[int],
        token_idx : List[int],
        top_k : int = 1,
        layers : List[int] = [-1],
        do_log_probs : bool = False,
        top_p : float = 1.0
    ) -> LogitResult:
        """
        Decode logits to tokens, after scoring top_k.
        NOTE: layers will index the literal layer index, not the index in the list. This means
        -1 corresponds to the last layer model_n_layer - 1.
        """
        # following will throw error if index is not present in collected layer logits:
        layers = [self._layer_idx.index(pos_indexing(i, n=self.n_layers)) for i in layers]
        # get ind
        token_idx = [pos_indexing(i, self._logits.shape[2]) for i in token_idx]

        # select logits while maintaining len(shape) = 4
        logits = self._logits[layers][:,prompt_idx][:,:,token_idx]
        
        if self.custom_decoder is not None:
            logits = self.custom_decoder(logits)
        
        logits = top_k_top_p_filtering(logits, top_k, top_p, do_log_probs)
        return LogitResult(logits.indices, logits.values)

def collect_hidden_states(
    model : LanguageModel,
    prompts : List[str],
    layers : List[int] = None,
    target_fn : Callable = None,
) -> List[torch.Tensor]:
    """
    Collect hidden states for each prompt. 
    Optionally, collect hidden states at specific layers, or at certain
    token positions given by target_fn.
    """
    if layers is None:
        layers = list(range(len(model.transformer.h)))

    with model.trace() as tracer:
        with tracer.invoke(prompts) as invoker:
            hidden_states = []
            for layer_idx in layers:
                # output is a tuple of (hidden_states, present)
                hs = model.transformer.h[layer_idx].output[0]
                if target_fn:
                    # mask shape: [n_prompts, n_toks]
                    mask = target_fn(invoker.inputs[0]["input_ids"])
                    hs = masked_get(hs, mask)
                hidden_states.append(hs)

        hidden_states = torch.stack(hidden_states, dim=0).save() 
            
    hidden_states = util.apply(hidden_states, lambda x: x.value, Proxy)
    return hidden_states

@torch.no_grad
def insert_patch(
    model : LanguageModel,
    prompts : List[str],
    patch : torch.Tensor, # [n_layer, n_prompt, n_tokens, hdim]
    layers_to_patch : List[int],
    target_fn : Callable,
    collect_hidden_states : bool = True
) -> TraceResult:
    """
    Insert patch at layers and tokens
    patch should have shape [model_n_layer, num_prompts, num_tokens_to_patch, n_embd]
    """
    if collect_hidden_states:
        collect_range = list(range(len(model.transformer.h)))
    else:
        collect_range = [-1]

    patch = patch.to(model.device)
    with model.trace() as tracer:
        with tracer.invoke(prompts) as invoker:
            inputs = invoker.inputs[0]["input_ids"]
            mask = target_fn(inputs)
            hidden_states = []
            for layer in range(len(model.transformer.h)):
                hs = model.transformer.h[layer].output[0]
                
                if layer in layers_to_patch:
                    new_hs = masked_fill(
                        hs, mask.to(hs.device), patch[layer,:]
                    )
                    for prompt_idx in range(hs.shape[0]):
                        model.transformer.h[layer].output[0][prompt_idx,:,:] = new_hs[prompt_idx,:]

                    # for i in range(hs.shape[0]):
                    #     model.transformer.h[layer].output[0][i,:,:] = patch[layer,i,:]

                if layer in collect_range:
                    hidden_states.append(model.transformer.h[layer].output[0])

            hidden_states = torch.stack(hidden_states,dim=0).save()
            logits = lm_decode(model, hidden_states, True).save()
            final_logits = model.lm_head.output.save()
    
    hidden_states = util.apply(hidden_states, lambda x: x.value, Proxy)
    logits = util.apply(logits, lambda x: x.value, Proxy)
    final_logits = util.apply(final_logits, lambda x: x.value, Proxy)
    assert torch.equal(logits[-1], final_logits), f"{logits[-1]} != {final_logits}"
    return TraceResult(logits, collect_range, len(model.transformer.h), hidden_states=hidden_states)
    
            
def logit_lens(
    model : LanguageModel,
    prompts : List[str],
    layers : List[int] = None,
    store_hidden_states : bool = False
) -> TraceResult:
    """
    Apply logit lens to prompts
    """
    if layers is None:
        layers = list(range(len(model.transformer.h)))
    else:
        layers = [pos_indexing(x, len(model.transformer.h)) for x in layers]
    
    with model.trace() as tracer:
        with tracer.invoke(prompts) as invoker:
            hidden_states = torch.stack([
                model.transformer.h[layer_idx].output[0]
                for layer_idx in layers
            ],dim=0).save()
            
            logits = lm_decode(model, hidden_states, True).save()
            final_logits = model.lm_head.output.save()

    logits = util.apply(logits, lambda x: x.value, Proxy)
    final_logits = util.apply(final_logits, lambda x: x.value, Proxy)
    assert torch.equal(logits[-1], final_logits)
    
    if store_hidden_states:
        hidden_states = util.apply(hidden_states, lambda x: x.value, Proxy)
        return TraceResult(logits, layers, len(model.transformer.h), hidden_states=hidden_states)
    else: 
        return TraceResult(logits, layers, model_n_layer=len(model.transformer.h))
    

def custom_lens(
    model : LanguageModel,
    decoder : torch.nn.Module,
    prompts : List[str],
    layers : List[int],
    activations : torch.Tensor = None,
    k : int = 1
) -> List[str]:
    """
    Apply custom lens to model activations for prompt at (layer, token)
    Note: for original decoder, load transformer version of model and copy
    """
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