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
from transformers import PreTrainedTokenizer
from typing import List
import sys
import os
from codetrace.interp_utils import *
from codetrace.interp_vis import *
import torch
from tqdm import tqdm
from collections import Counter
import random

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
        hidden_states.append(hs.mean(dim=1)) # batch size mean
        
    # save tensor
    hidden_states = torch.cat(hidden_states, dim=1)
    print(f"Hidden states shape before avg: {hidden_states.shape}")
    return hidden_states.mean(dim=1)


def filter_prompts(prompts : List[str], 
                   labels : List[str],
                   hexshas : List[str],
                   single_tokenize : PreTrainedTokenizer = None,
                   dedup_threshold : int = 3,
                   dup_typ_threshold : int = 10) -> List[Tuple[str, str, str]]:
    """
    Balance prompts s.t. there is a balanced distribution of labels.
    Do not use more than max_size prompts.
    Remove multi-token label prompts if tokenizer is passed.
    Deduplicate prompts by hexsha by some dedup_threshold (max prompts for a program)
    """
    assert len(prompts) == len(labels) == len(hexshas)
    if not single_tokenize is None:
        prompts = [p for i,p in enumerate(prompts) if len(single_tokenize.encode(labels[i])) == 1]
        labels = [l for l in labels if len(single_tokenize.encode(l)) == 1]
        
    # get count of labels
    counter = Counter(labels)
    
    hexsha_count = {hexsha : 0 for hexsha in hexshas}
    label_count = {label : 0 for label in counter.keys()}
    balanced_prompts = []
    for i,p in enumerate(prompts):
        if label_count[labels[i]] > dup_typ_threshold:
            continue
        if hexsha_count[hexshas[i]] > dedup_threshold: # some threshold
            continue
        balanced_prompts.append((p, labels[i], hexshas[i]))
        label_count[labels[i]] += 1
        hexsha_count[hexshas[i]] += 1


    return balanced_prompts


if __name__ == "__main__":
    # test get_averages
    model = LanguageModel("/home/arjun/models/starcoderbase-1b")
    ds = datasets.load_dataset("franlucc/starcoderbase-1b-completions_typeinf_analysis", split="train")
    ds = ds.filter(lambda x : x["correctness"] == "correct")
    # test balance prompts
    prompts = ds['prompt']
    labels = ds['solution']
    hexshas = ds['hexsha']
    out = filter_prompts(prompts, labels, hexshas, dedup_threshold=4,single_tokenize=model.tokenizer)
    df = pd.DataFrame(out, columns=["prompt", "label", "hexsha"])
    # count number of labels, hexshas
    def pretty_print(x):
        # print a df column in full
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(x)
    pretty_print(df['label'].value_counts())
    pretty_print(df['hexsha'].value_counts())
    print(df.shape)