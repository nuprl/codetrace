import os
import datasets
import argparse
from typing import List, Tuple, Dict, Any
from codetrace.batched_utils import batched_collect_activations, batched_patch
from codetrace.utils import (
    mask_target_idx, 
    load_dataset, 
    HiddenStateStack_1tok, 
    masked_add, 
    reset_index_dim0, 
    get_lm_layers
)
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from nnsight import LanguageModel
import itertools as it
import einops

def _chain_activations(activations: List[HiddenStateStack_1tok], layer:int) -> np.ndarray:
    numpy_activ = []
    while len(activations) > 0:
        activ = activations.pop()[:,layer].squeeze() # nsamples, nprompt, hdim
        numpy_activ.append(activ)
    numpy_activ = torch.stack(numpy_activ, dim=0).to(torch.float32)
    return einops.rearrange(numpy_activ, "num_cases n_samples n_prompt hdim -> (num_cases n_samples n_prompt) hdim")

def _tensorname(path: str) -> str:
    assert "steering" in path
    names = path.split("/")
    for n in names:
        if "steering" in n:
            return n + ".pt"

def plot_tsne(
    labeled_activations: List[Tuple[datasets.Dataset, torch.Tensor]], 
    layers: List[int],
    outfile: str
):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    pca = PCA(n_components=50, random_state=42)

    num_layers = len(layers)
    fig, axes = plt.subplots(1, num_layers, figsize=(25, 25), constrained_layout=True)
    
    if num_layers == 1:
        axes = [axes]

    colors = ['red', 'blue', 'green']

    for idx, layer in enumerate(layers):
        ax = axes[idx]
        data = _chain_activations([a for _, a in labeled_activations], layer)
        data_pca = pca.fit_transform(data)
        data_tsne = tsne.fit_transform(data_pca)

        print(data_tsne.shape)
        n = 0
        for i,(label, activations) in enumerate(labeled_activations):
            interval = activations.shape[0]*activations.shape[2]
            indices = list(range(n, n+interval)) # nsamples, nprompt
            assert len(indices) != []
            n += interval
            color = colors[i]
            print(color, label, layer, min(indices), max(indices))
            ax.scatter(
                data_tsne[indices, 0], data_tsne[indices, 1], 
                c=color, label=f'Class {_tensorname(label)}', alpha=0.6
            )
        
        ax.set_title(f't-SNE Projection (Layer {layer})')
        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
        ax.grid(True)
        ax.legend()

    plt.savefig(outfile)


def main(
    model: str,
    datasets: List[Path],
    layers: List[int],
    field: str,
    outfile: str,
    cache_dir: str,
    batch_size: int = 5,
    patches: List[str] = None
):
    for i in range(len(datasets)):
        ds = load_dataset(datasets[i])
        ds._name = datasets[i]
        datasets[i] = ds

    model = LanguageModel(model, device_map="cuda", torch_dtype=torch.bfloat16, dispatch=True)
    labeled_activations = []
    for sample,ds in enumerate(datasets):
        activ_path = f"{cache_dir}/{_tensorname(ds._name)}"
        print("--DATASET NAME--:", ds._name, activ_path)
        assert len(ds) % batch_size == 0, "Dataset must be divisible by batch_size"

        if Path(activ_path).exists():
            activ = torch.load(activ_path)
        elif patches:
            patch = torch.load(patches[sample])
            activ_list = batched_patch(
                model, 
                list(ds[field]),
                patch,
                layers,
                lambda x: mask_target_idx(x, [-1]),
                masked_add,
                reduction="sum",
                batch_size=batch_size,
                collect_hidden_states=layers
            )
            activ = torch.stack(activ_list, dim=0)
            torch.save(activ, activ_path)
        else:
            activ_list = batched_collect_activations(
                model, 
                list(ds[field]),
                lambda x: mask_target_idx(x, [-1]),
                reduction="sum",
                layers=layers,
                batch_size=batch_size
            )
            activ = torch.stack(activ_list, dim=0)
            torch.save(activ, activ_path)

        labeled_activations.append((ds._name, activ))
        print(activ.shape)
    
    min_n_sample = min([a.shape[0] for _,a in labeled_activations])
    labeled_activations = [(l,a[:min_n_sample]) for l,a in labeled_activations]
    plot_tsne(labeled_activations, layers, outfile)
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--layers", required=True)
    parser.add_argument("--outfile", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--field", default="mutated_program")
    parser.add_argument("--patches", type=str, default="None")
    parser.add_argument("--batch-size", type=int, default=5)

    args = parser.parse_args()
    args.layers = [int(l) for l in args.layers.split(",")]
    args.datasets = args.datasets.split(",")
    if args.patches == "None":
        args.patches = None
    else:
        args.patches = args.patches.split(",")
    os.makedirs(args.cache_dir, exist_ok=True)
    print("DATASETS:",args.datasets)
    main(**vars(args))
    
    
