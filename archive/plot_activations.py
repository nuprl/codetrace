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
    # n_layer, n_samples, hdim
    activ = torch.concat(activations, dim=1).to(torch.float32)[layer].squeeze()
    return activ.numpy() # n_samples, hdim

def _tensorname(path: str) -> str:
    assert "steering" in path
    names = path.split("/")
    for n in names:
        if "steering" in n:
            return n + ".pt"

def plot_tsne(
    labeled_activations: List[Tuple[datasets.Dataset, torch.Tensor, str]], 
    layers: List[int],
    outfile: str,
    random_state: int = None
):
    tsne = TSNE(n_components=2, perplexity=30, random_state=random_state)
    pca = PCA(n_components=50, random_state=random_state)

    num_layers = len(layers)
    fig, axes = plt.subplots(len(labeled_activations) // 2, num_layers, figsize=(15, 10), constrained_layout=True)
    
    if num_layers == 1:
        axes = [axes]

    colors = ['red', 'blue', 'green', 'yellow', 'orange', 'cyan', 'black']

    for j,activs in enumerate(zip(labeled_activations[::2], labeled_activations[1::2])):
        for idx, layer in enumerate(layers):
            ax = axes[j%(len(labeled_activations) // 2), idx]
            data = _chain_activations([a for _, a,_ in activs], layer)
            data_pca = pca.fit_transform(data)
            data_tsne = tsne.fit_transform(data_pca)

            print(data_tsne.shape)
            n = 0
            for i,(label, activations, succ_label) in enumerate(activs):
                interval = activations.shape[1]
                indices = list(range(n, n+interval)) # nsamples, nprompt
                assert len(indices) != []
                n += interval
                color = colors[i]
                print(color, label, layer, min(indices), max(indices))
                ax.scatter(
                    data_tsne[indices, 0], data_tsne[indices, 1], 
                    c=color, label=f'Class {_tensorname(label).replace(".pt", succ_label)}', alpha=0.6
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
        ds = ds.map(lambda x: {**x, "success":x["steered_predictions"] == x["fim_type"]})
        ds._name = datasets[i]
        datasets[i] = ds

    model = LanguageModel(model, device_map="cuda", torch_dtype=torch.bfloat16, dispatch=True)
    labeled_activations = []
    for sample,dataset in enumerate(datasets):
        for succ_label in [True, False]:
            ds = dataset.filter(lambda x: x["success"] == succ_label)
            succ_label = "succ" if succ_label else "fail"
            activ_path = f"{cache_dir}/{_tensorname(dataset._name).replace('.pt',succ_label + '.pt')}"
            print("--DATASET NAME--:", activ_path)

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
                activ = torch.concat(activ_list, dim=1)
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
                activ = torch.concat(activ_list, dim=1)
                torch.save(activ, activ_path)

            labeled_activations.append((dataset._name, activ, succ_label))
            print(activ.shape)
    
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
    
    
