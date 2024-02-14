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

def patch_logit_heatmap(model : LanguageModel,
                        clean_prompt : List[str] | str, 
                        corrupted_prompt : List[str] | str,
                        patched_logits : TraceResult,
                        layers_patched : List[int],
                        clean_token_idx : int,
                        corrupted_token_idx : int, 
                        figsize : Tuple[int,int] = (10,10),
                        figtitle : str = "Patch Logit Heatmap",):
    """
    Given hidden states from a patch clean[j]->corrupted[j] at layers,
    do a heatmap where x is layers, y is original top token logit at (layer, i) in corrupted
    and color is patched top token logit at (layer, i) in patched.
    Legend stores the patched in token from clean[j]
    Store original clean prediction and corrupted predicion?
    """
    if isinstance(clean_prompt, List) or isinstance(corrupted_prompt, List):
        raise NotImplementedError("Not implemented for multiple prompts")
    
    plt.figure(figsize=figsize)
    x_axis_labels = [f"Layer {i}" for i in layers_patched]
    y_len = len(clean_prompt) if isinstance(clean_prompt, List) else 1
    y_axis_labels = [f"Example {i}" for i in range(y_len)]
    
    plt.yticks(list(range(len(y_axis_labels))), y_axis_labels)
    plt.xticks(list(range(len(x_axis_labels))), x_axis_labels)
    plt.xlabel("Patched Layer")
    plt.title(figtitle)

    patched_logits = patched_logits.score_top(1)
    patched_top_idx = patched_logits.get_indexes(layers_patched, corrupted_token_idx)
    patched_top_tokens = patched_logits.get_tokens(model, layers=layers_patched, token_idx=corrupted_token_idx)
    probabilities = patched_logits.get_probabilities(layers=layers_patched, token_idx=corrupted_token_idx, do_log_probs=False).cpu().flatten().numpy()
    
    probabilities = probabilities.reshape(y_len, len(layers_patched))
    im = plt.imshow(probabilities, cmap="viridis", aspect="auto")
    
    # color is the probability of patched token
    color_map = im.cmap(probabilities)
    # add colorbar
    plt.colorbar(im)

    # use map to make legend
    original_corrupted_logits = logit_lens(model, [corrupted_prompt])
    corrupt_orig_pred = original_corrupted_logits.score_top(1).get_tokens(model, layers=layers_patched, token_idx=corrupted_token_idx)

    # get target clean tokens
    original_clean_logits = logit_lens(model, [clean_prompt])
    clean_token_pred = original_clean_logits.score_top(1).get_tokens(model, layers=layers_patched, token_idx=clean_token_idx)
    
    # for each probability square, legend shows relative clean_tok->corr_tok
    patches = []
    for i,(a,b) in enumerate(list(zip(clean_token_pred, corrupt_orig_pred))):
        patches.append(mpatches.Patch(color='grey', label=f"{layers_patched[i]}:'{a}' -> '{b}'"))
    plt.legend(handles=patches, loc="upper right", bbox_to_anchor=(1.5, 1.0), title="layer:clean->corrupt")
    
    # build an annotations dict for each square in grid with values from probs_patched_results
    annotations = {i:{} for i in range(len(x_axis_labels)*len(y_axis_labels))}
    for i in range(len(x_axis_labels)):
        for j in range(len(y_axis_labels)):
            probs = probabilities[j][i]
            annotations[i+len(y_axis_labels)*j] = {"toptok": f"'{patched_top_tokens[i]}'","prob": f"{probs:.2f}"}
            
    # create tuples of positions
    positions =[(x , y ) for x in range(len(x_axis_labels)) for y in range(len(y_axis_labels))]

    # add annotations
    for pos, text in annotations.items():
        plt.annotate(text["toptok"], xy=positions[pos],color="black", fontsize=12, ha="center", va="center")

    plt.tight_layout()
    plt.savefig(figtitle.lower().replace(" ", "_") + ".png")