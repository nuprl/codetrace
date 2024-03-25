from ast import literal_eval
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sys
import os
import json
from glob import glob
import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from typing import List
import numpy as np
import datasets


finetune_starcoder_py_results = [
    {"checkpoint":"0","humaneval": 0.14161490683229813, "accuracy":0, "marker":"."},
    {"checkpoint":"1","humaneval": 0.04440993788819876, "accuracy":0.86, "marker":"."},
    {"checkpoint":"2","humaneval": 0.08167701863354036, "accuracy":0.93, "marker":"."},
    {"checkpoint":"3", "humaneval": 0.04534161490683229, "accuracy":0.95, "marker":"."},
    {"checkpoint":"4","humaneval": 0.0686335403726708, "accuracy":0.93, "marker":"."},
    {"checkpoint":"5","humaneval": 0.049689440993788817, "accuracy":0.91, "marker":"."},
    {"checkpoint":"6","humaneval": 0.04751552795031056, "accuracy":0.96, "marker":"."},
]

def marker_for_len(l):
    if l == 1:
        return "."
    elif l == 3:
        return "o"
    else:
        return "x"
    
def layer_sort(df, layer_n, window_size):
    all_layers = list(range(layer_n))
    zipped = list(zip([all_layers[i:i+window_size] for i in all_layers]))
    windows = [str(j[0]) for j in zipped if len(j[0]) == window_size]

    # add column with sort_idx
    df["sort_idx"] = df["layers_patched"].map(lambda x: windows.index(x))
    df["marker"] = df["lengths"].map(marker_for_len)
    # sort grouped by window key
    df = df.sort_values(by="sort_idx")
    # set sort_idx as new index
    df.set_index("sort_idx", inplace=True)
    df["layers_patched"] = df["layers_patched"].map(lambda x: str(x).replace("]","").replace("[","").replace(",","-"))
    return df

def counter(data, all_labels):
    count = {d:0 for d in all_labels}
    for i in data:
        if i in count.keys():
            count[i] += 1
    return count

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def kl(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    return np.sum(np.where(p != 0, p * np.log(p / q), 0))
 
def divergence(ds):
    all_types = ds["fim_type"]
    x = softmax(list(counter(ds["fim_type"], all_types).values()))
    y = softmax(list(counter(ds["steered_generation"], all_types).values()))
    return kl(x,y)


def get_kl_div(df, parent_dir):
    df["results_ds"] = df["experiment_dir"].map(lambda x: f"{parent_dir}/{x}/ood_steering_results_ds")
    kl_divs = {}
    skews = {}
    for path in df["results_ds"]:
        ds = datasets.load_from_disk(path)
        kl = divergence(ds)
        kl_divs[path] = kl
        skews[path] = skew(ds)
    max_kl = max(kl_divs.values())
    max_skew = max(skews.values())
    df["kl_div"] = df["results_ds"].map(lambda x: kl_divs[x] / max_kl)
    df["skew"] = df["results_ds"].map(lambda x: skews[x] / max_skew)
    return df, max_kl, max_skew


def plot_layer_ablation(data: List[pd.DataFrame], outfile, parent_dir = None):
    """
    Figure, 3 subplots (for window size 1,3,5)
    - x: layers patched
    - y: accuracy on ood
    """
    plt.figure(figsize=(20,4))
    n_layer = len(data[0]["layers_patched"].unique())
    lengths = [1,3,5]
    for i,df in enumerate(data, start=1):
        # add a kl_divergence column
        df, max_kl, max_skew = get_kl_div(df, parent_dir)
        df = layer_sort(df, n_layer, lengths[i-1])
        subplot = f"1{len(data)}{i}"
        ax = plt.subplot(int(subplot))
        sns.lineplot(data=df, x="layers_patched", y="ood_accuracy", markers=True, style="marker",linewidth=2)
        sns.lineplot(data=df, x="layers_patched", y="kl_div", markers=True, style="marker",linewidth=2, color="red")
        y = df["ood_accuracy"]
        for xy in zip(list(range(len(y))), y):
            text = xy[1]
            ax.annotate(f'{text:.2f}',xy,va="bottom", ha="right", fontsize=8, color="blue")
        y = df["kl_div"] 
        for xy in zip(list(range(len(y))), y):
            text = xy[1] * max_kl
            ax.annotate(f'{text:.2f}',xy,va="bottom", ha="right", fontsize=8, color="red")
        
        ax.set_ylim(0,1)
        ax.legend()
        plt.xticks(rotation=45, ha="right")
        ax.set_xlabel("Layers Patched", fontsize=12)
        ax.set_ylabel("Accuracy on Held out set", fontsize=12)

    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()


def plot_finetuning_ablation(df: pd.DataFrame, outfile):
    plt.figure(figsize=(12,7))
    ax = plt.subplot()
    sns.lineplot(data=df, x="checkpoint", y="humaneval",markers=True, style="marker",linewidth=2)
    sns.lineplot(data=df, x="checkpoint", y="accuracy",markers=True, style="marker",linewidth=2)
    
    for item in zip(df["checkpoint"], df["humaneval"]):
        x = item[0]
        y = item[1]
        ax.text(x,y,f'{y:.2f}',color="blue", va="bottom", linespacing=2, fontsize=12)

    for item in zip(df["checkpoint"], df["accuracy"]):
        x = item[0]
        y = item[1]
        ax.text(x,y,f'{y:.2f}',color="orange", va="bottom", ha="right", linespacing=2, fontsize=12)
    
    plt.ylim(0,1)
    ax.legend()
    ax.set_xlabel("Epoch number", fontsize=12)
    ax.set_ylabel("", fontsize=12)
    custom_handles = [Line2D([0], [0], color='orange', lw=1, label="Accuracy on Held out set"),
                      Line2D([0], [0], color='blue', lw=1, label="Humaneval performance")]
    
    custom_legend = plt.legend(handles=custom_handles, loc='center right', title="Legend", fontsize=12)
    ax.add_artist(custom_legend)
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()
    
def plot_main_results(df, outfile, accuracy_kind="ood_accuracy"):
    """
    Plots main results figure.
    
    Needs:
    - model name
    - python/ts performance on each mut dataset
    - python/ts rand tensor performance
    """
    bars_label_order = ["1", "2", "3",
                        "1 & 2", "1 & 3", "2 & 3", 
                        "1 & 2 & 3"]
    
    def map_to_label(colname, lang):
        dikt = {
            f"{lang}_all_mutations": "1 & 2 & 3",
            f"{lang}_rename_var": "1",
            f"{lang}_rename_types": "2",
            f"{lang}_delete_fit": "3",
            f"{lang}_vars_and_delete": "1 & 3",
            f"{lang}_types_and_delete": "2 & 3",
            f"{lang}_all_renames": "1 & 2"
        }
        key = [k for k in dikt.keys() if colname.startswith(k)]
        if not key:
            return "NONE"
        else:
            key = key[0]
            
        if "with" in colname and "tensor" in colname:
            return "NONE"
        
        return dikt[key]
    
    df_py = df[df["experiment_dir"].str.startswith("py_")]
    df_py["rand"] = df_py["experiment_dir"].str.contains("rand")
    df_ts = df[df["experiment_dir"].str.startswith("ts_")]
    df_ts["rand"] = df_ts["experiment_dir"].str.contains("rand")
    
    df_ts_rand = df_ts[df_ts["experiment_dir"].str.contains("rand")]
    df_py["experiment_dir"] = df_py["experiment_dir"].map(lambda x: map_to_label(x, "py"))
    df_ts["experiment_dir"] = df_ts["experiment_dir"].map(lambda x: map_to_label(x, "ts"))
    df_py = df_py[df_py["experiment_dir"] != "NONE"]
    df_ts = df_ts[df_ts["experiment_dir"] != "NONE"]
    
    steer_py = df_py[df_py["rand"] == False]
    rand_py = df_py[df_py["rand"] == True]
    steer_ts = df_ts[df_ts["rand"] == False]
    rand_ts = df_ts[df_ts["rand"] == True]
    
    def _reindex(df):
        df = df.set_index('experiment_dir')
        return df.reindex(bars_label_order).reset_index()
    
    steer_py = _reindex(steer_py)
    rand_py = _reindex(rand_py)
    steer_ts = _reindex(steer_ts)
    rand_ts = _reindex(rand_ts)
    
    accuracy = accuracy_kind
    
    plt.figure(figsize=(12,7))
    ax = plt.subplot()
    width = 0.35
    x = np.arange(len(bars_label_order))
    
    ax.bar(x - width/2, steer_py[accuracy_kind], width=width, color="cornflowerblue", edgecolor="black", label="Python", alpha=0.7)
    ax.bar(x - width/2, rand_py[accuracy_kind], width=width, color="cornflowerblue", hatch="//", edgecolor="black", label="Python Random Baseline", alpha=0.7)
    
    ax.bar(x + width/2, steer_ts[accuracy_kind], width=width, color="darkorange", edgecolor="black", label="TypeScript", alpha=0.7)
    ax.bar(x + width/2, rand_ts[accuracy_kind], width=width, color="darkorange", hatch="//", edgecolor="black", label="TypeScript Random Baseline", alpha=0.7)
    
    # Add text annotations above bars
    for p in ax.patches:
        ax.annotate(str(round(p.get_height(), 2)), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    
    custom_legend_labels = {"1": "1: Rename variables", "2": "2: Rename types", "3": "3: Remove type annotations"}
    custom_handles = [Line2D([0], [0], color='white', lw=1, label=custom_legend_labels['1']),
                      Line2D([0], [0], color='white', lw=1, label=custom_legend_labels['2']),
                      Line2D([0], [0], color='white', lw=1, label=custom_legend_labels['3'])]
    
    custom_legend = plt.legend(handles=custom_handles, loc='upper right', title="Mutations")
    ax.add_artist(custom_legend)
    
    plt.ylim(0,1)
    ax.set_xlabel("Mutation Dataset", fontsize=12)
    ax.set_ylabel("Accuracy on Held-out Evaluation Set", fontsize=12)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, ncol=4, fontsize=12)
    plt.tight_layout()
    ax.set_xticks(x)
    ax.set_xticklabels(bars_label_order)
    plt.savefig(outfile)
    plt.close()

    
parser = argparse.ArgumentParser()
parser.add_argument("--outfile", type=str, required=True)
parser.add_argument("--results_csv", type=str, nargs="+", required=False)
parser.add_argument("--plotfunc", type=str, required=True, choices=["main", "ftune", "layer"])
parser.add_argument("--parent_dir", type=str, required=False)
args = parser.parse_args()

if args.plotfunc == "main":
    data = []
    for f in list(args.results_csv):
        df = pd.read_csv(f)
        data.append(df)
        
    df = pd.concat(data).reset_index()
    # print(df)
    plot_main_results(df, args.outfile, "ood_accuracy")
    
if args.plotfunc == "ftune":
    plot_finetuning_ablation(pd.DataFrame(finetune_starcoder_py_results), args.outfile)
    
if args.plotfunc == "layer":
    df = pd.read_csv(args.results_csv[0])
    # serapate into 3
    df["lengths"] = df["layers_patched"].map(lambda x: len(str(x).split(",")))
    df0 = df.loc[df["lengths"] == 1]
    df1 = df.loc[df["lengths"] == 3]
    df2 = df.loc[df["lengths"] == 5]
    plot_layer_ablation([df0,df1,df2], args.outfile, args.parent_dir)

