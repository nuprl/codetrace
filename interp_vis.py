"""
Interp visualizations
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
from interp_utils import *
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import matplotlib as mpl

def patched_heatmap_prediction(model : LanguageModel,
                        clean_prompts : List[str] | str, 
                        corrupted_prompts : List[str] | str,
                        patched_final_logits : List[TraceResult],
                        layers_patched : List[int],
                        clean_token_idx : int = -1,
                        corrupted_token_idx : int = -1, 
                        annotations : List[Tuple[str,str]] | None = None,
                        figsize : Tuple[int,int] = (10,10),
                        bbox_to_anchor : Tuple[int,int] = (1.6, 1.0),
                        figtitle : str = "Patch Logit-Prediction Heatmap",
                        outfile : str = "temp.png"):
    """
    Cells on heatmap should be the final layer top logit that changes
    after patch.
    
    All prompt pairs are patched in the same layers. Patched logits is
    list of trace results of the final layer, one for each prompt pair.
    
    NOTE: 
    - assumes top_k in logits is 1
    - patched_final_logits is 2D when there are multiple prompt pairs, else 1D
        - 1D is one TraceResult for each layer
        - 2D is a list of TraceResults for each layer.
            ex: patched_final_logits[0] is results for each prompt at layer layers_patched[0]
    """
    clean_prompts = arg_to_list(clean_prompts)
    corrupted_prompts = arg_to_list(corrupted_prompts)
    if not isinstance(patched_final_logits, list):
        raise ValueError("Given TraceResult, expected List.\nDid you forget to do an individual patch for each layer?")
    
    plt.figure(figsize=figsize)
    x_len = len(layers_patched)
    x_axis_labels = [f"Layer {i}" for i in layers_patched]
    y_len = len(clean_prompts)
    y_axis_labels = [f"Prompt {i}" for i in range(y_len)]
    
    plt.yticks(list(range(len(y_axis_labels))), y_axis_labels)
    plt.xticks(list(range(len(x_axis_labels))), x_axis_labels)
    plt.xlabel("Patched Layer")
    plt.title(figtitle)

    # need token and probabilities
    tokens, probs = [], []
    for layer_patch in patched_final_logits:
        patched_logits : LogitResult = layer_patch.decode_logits(prompt_idx=list(range(y_len)))
        pt, pp = [], []
        for i in range(y_len):
            pt.append(np.array(patched_logits[0][i].tokens(model.tokenizer)).item())
            pp.append(patched_logits[0][i].probs().item())
        tokens.append(pt)
        probs.append(pp)
        
    tokens = np.array(tokens)
    probs = np.array(probs)
    
    cmap = mpl.cm.cool
    im = plt.imshow(probs.transpose(), cmap=cmap, aspect="auto")
    
    # color is the probability of patched token
    im.cmap(probs)
    # add colorbar
    plt.colorbar(im)

    # legend is clean prediction -> corrupted prediction
    patches = []
    if not annotations:
        def original_prompt_pred(model, prompts):
            logits = logit_lens(model, prompts).decode_logits(prompt_idx=list(range(y_len)))
            tokens, probs = [], []
            for i in range(len(prompts)):
                t = logits[0][i].tokens(model.tokenizer)
                p = logits[0][i].probs()
                tokens.append(t)
                probs.append(p)
            return np.array(tokens), probs
        
        original_clean = original_prompt_pred(model, clean_prompts)
        original_corrupted = original_prompt_pred(model, corrupted_prompts)

        # for each probability square, legend shows relative clean_tok->corr_tok
        patches = []
        for i in range(y_len):
            a_pred = original_clean[0][i].item()
            a_prob = float(f"{original_clean[1][i].item():.2f}")
            b_pred = original_corrupted[0][i].item()
            b_prob = float(f"{original_corrupted[1][i].item():.2f}")
            a = (a_pred, a_prob)
            b = (b_pred, b_prob)
            patches.append(mpatches.Patch(color='grey', label=f"{i}: {a} (->) {b}"))
    else:
        # NOTE: probs not supported for annotations
        for i,(a,b) in enumerate(annotations):
            patches.append(mpatches.Patch(color='grey', label=f"{i}: {repr(a)} (->) {repr(b)}"))
        
    plt.legend(handles=patches, loc="center right", bbox_to_anchor=bbox_to_anchor, title="original predictions\nprompt: clean (->) corrupt")

    # build an annotations dict for each square in grid with values from probs_patched_results
    annotations = {}
    for i in range(x_len):
        for j in range(y_len):
            p = probs[i][j].item()
            annotations[(i,j)] = {"toptok": f"{tokens[i][j].item()}","prob": f"{p:.2f}"}
    
    # add annotations
    for pos, text in annotations.items():
        plt.annotate(text["toptok"], xy=pos,color="black", fontsize=10, ha="center", va="center")

    plt.tight_layout()
    plt.savefig(outfile)
    