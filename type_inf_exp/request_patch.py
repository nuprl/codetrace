"""
In this file, we try to find a model edit at the representation level that
is able to fix broken type-inf predictions. Use the following methods

Method 1.
    Make an average of <fim_prefix>, <fim_suffix> and <fim_middle> tokens
    at all layers of correct runs. [X]
    Patch these in before layer 14 (corresponding layer only).
        a. substitute layers (0 to 14) with the corresponding average at FIM tokens only
        b. substitute layers (0 to 14) with the corresponding average at all tokens
        c. add the corresponding average at FIM tokens only at layers (0 to 14)
        d. add the corresponding average at all tokens at layers (0 to 14)
        e. do above but with less layers / samples for avg
    
Method 2.
    Make an average of the entire token sequence at all layers of correct runs.
    Vary number of samples and layers. Patch these in before layer 14 (corresponding layer only).
    
Method 3.
    Do Methdo 1,2 but with sampling instead of greedy predictions. Do beam search.

NOTE: if all else fails, token indexing may be wrong.
ALso -> try some ICL. A system prompt "Infer the type"
"""
from nnsight import LanguageModel
from typing import List
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from interp_utils import *
from interp_vis import *
import torch
from tqdm import tqdm

def get_averages(model: LanguageModel,
                 prompts : List[str],
                 tokens : List[str],
                 batch_Size=5) -> torch.Tensor:
    """
    Get averages of tokens at all layers for all prompts
    
    NOTE: if tokens aren't unique in the prompts, the first occurence is used.
    """
    tokens = [model.tokenizer.encode(t)[0] for t in tokens]
    
    # batch prompts according to batch size
    prompt_batches = [prompts[i:i+batch_Size] for i in range(0, len(prompts), batch_Size)]
    hidden_states = []
    for batch in tqdm(prompt_batches, desc="Batch"):
        hs = collect_hidden_states_at_tokens(model, batch, tokens)
        hidden_states.append(hs.mean(dim=1))
        
    # save tensor
    hidden_states = torch.cat(hidden_states, dim=1)
    print(f"Hidden states shape before avg: {hidden_states.shape}")
    return hidden_states.mean(dim=1)



if __name__ == "__main__":
    # test get_averages
    model = LanguageModel("/home/arjun/models/starcoderbase-7b")
    ds = datasets.load_dataset("franlucc/stenotype-eval-dataset-func-type-stripped-v4", split="train")
    prompts = [placeholder_to_std_fmt(ex["fim_program"], STARCODER_FIM) for ex in ds][:10]
    tokens = ["<fim_prefix>", "<fim_suffix>", "<fim_middle>"]
    get_averages(model, prompts, tokens)