from nnsight import LanguageModel, util
from nnsight.tracing.Proxy import Proxy
from typing import *
import torch
import circuitsvis as cv
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def greedy_logit_lens(model: LanguageModel,
               prompt: str,
               layers: List[int] = None,
               token_idxs: List[int] = [-1]) -> Dict[int, Dict[int, str]]:
    """
    Returns a dict of top logit for each layer and token index.
    """
    if layers is None:
        layers = list(range(model.config.n_layer))
        
    def greedy_decode(hs : torch.Tensor):
        return model.lm_head(model.transformer.ln_f(hs))
        
    logit_idx = {i: {} for i in layers}
    
    with model.invoke(prompt) as invoker:
        tokens = invoker.input.tokens()
        for l in layers:
            hidden_states = greedy_decode(model.transformer.h[l].output[0])
            for idx in token_idxs:
                hidden_states_id = hidden_states[0, idx, :]
                probs = hidden_states_id.softmax(dim=-1).argmax().save()
                logit_idx[l][idx] = probs
                    
    for l in layers:
        for idx in token_idxs:
            token_id = util.apply(logit_idx[l][idx], lambda x: x.item().value, Proxy)
            logit_idx[l][idx] = model.tokenizer.decode(token_id)
        
    return logit_idx, tokens


def top_logit_lens(model: LanguageModel,
               prompt: str,
               top_p : int = 10,
               layers: List[int] = None,
               token_idxs: List[int] = [-1]) -> Dict[int, Dict[int, str]]:
    """
    Returns a dict of top logit for each layer and token index.
    """
    if layers is None:
        layers = list(range(model.config.n_layer))
        
    def greedy_decode(hs : torch.Tensor):
        return model.lm_head(model.transformer.ln_f(hs))
        
    logit_idx = {i: {} for i in layers}
    
    with model.invoke(prompt) as invoker:
        tokens = invoker.input.tokens()
        for l in layers:
            hidden_states = greedy_decode(model.transformer.h[l].output[0])
            for idx in token_idxs:
                hidden_states_id = hidden_states[0, idx, :]
                # get top p logit ids with probabilities
                top_vals_and_idx = hidden_states_id.softmax(dim=-1).topk(top_p).save()
                logit_idx[l][idx] = top_vals_and_idx
                    
    for l in layers:
        for idx in token_idxs:
            token_ids_and_probs = util.apply(logit_idx[l][idx], lambda x: x.value, Proxy)
            values = token_ids_and_probs.values
            indices = token_ids_and_probs.indices
            indices = [model.tokenizer.decode(i) for i in indices]
            values = [v.cpu().item() for v in values]
            logit_idx[l][idx] = [(i, j) for i, j in zip(indices, values)]
        
    return logit_idx, tokens


def attention_vis(model: LanguageModel,
                       prompt: str,
                       layers: List[int] = None) -> List[torch.Tensor]:
    """
    Returns a list of attention patterns for each layer.
    """
    with model.generate(max_new_tokens=1) as generator:
        with generator.invoke(prompt) as invoker:
            tokens = invoker.input.tokens()
            attn_hidden_states = [model.transformer.h[layer_idx].attn.output[1][0].save() for layer_idx in range(len(model.transformer.h))]

    attn_hidden_states = util.apply(attn_hidden_states, lambda x : x.value, Proxy)
    
    return attn_hidden_states, tokens