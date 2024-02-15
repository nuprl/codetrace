"""
In this file, we try to find a model edit at the representation level that
is able to fix broken type-inf predictions. Use the following methods

Method 1.
    Make an average of <fim_prefix>, <fim_suffix> and <fim_middle> tokens
    at all layers of correct runs. Vary number of samples and layers. Patch
    these in before layer 14 (corresponding layer only).
    
Method 2.
    Make an average of the entire token sequence at all layers of correct runs.
    Vary number of samples and layers. Patch these in before layer 14 (corresponding layer only).
    
Method 3.
    Do Methdo 1,2 but with sampling instead of greedy predictions. Do beam search.
    
"""
from nnsight import LanguageModel
from typing import List
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from interp_utils import *
from interp_vis import *
import torch

def get_averages(model: LanguageModel,
                 prompts : List[str],
                 tokens : List[str],
                 batch_Size=5) -> torch.Tensor:
    """
    Get averages of tokens at all layers for all prompts
    """
    tokens = [model.tokenizer.encode(t).item() for t in tokens]
    
    # batch prompts according to batch size
    prompt_batches = [prompts[i:i+batch_Size] for i in range(0, len(prompts), batch_Size)]
    hidden_states = []
    for batch in prompt_batches:
        hs = collect_hidden_states_at_tokens(model, batch, tokens)
        hidden_states.append(hs)
        
    # save tensor
    hidden_states = torch.cat(hidden_states)
    return hidden_states.mean(dim=1)