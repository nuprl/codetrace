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
    {"checkpoint":"0","humaneval": 0.14161490683229813, "accuracy":0},
    {"checkpoint":"1","humaneval": 0.04440993788819876, "accuracy":0.86},
    {"checkpoint":"2","humaneval": 0.08167701863354036, "accuracy":0.93},
    {"checkpoint":"3", "humaneval": 0.04534161490683229, "accuracy":0.95},
    {"checkpoint":"4","humaneval": 0.0686335403726708, "accuracy":0.93},
    {"checkpoint":"5","humaneval": 0.049689440993788817, "accuracy":0.91},
    # {"checkpoint":"6","humaneval": 0.04751552795031056, "accuracy":0.96},
]

def layer_sort(df, layer_n, window_size):
    all_layers = list(range(layer_n))
    zipped = list(zip([all_layers[i:i+window_size] for i in all_layers]))
    windows = [str(j[0]) for j in zipped if len(j[0]) == window_size]

    # add column with sort_idx
    df["sort_idx"] = df["layers_patched"].map(lambda x: windows.index(x))
    # sort grouped by window key
    df = df.sort_values(by="sort_idx")
    # set sort_idx as new index
    df.set_index("sort_idx", inplace=True)
    df["layers_patched"] = df["layers_patched"].map(lambda x: str(x).replace("]","").replace("[","").replace(",","-"))
    
    def rename(x):
        if "-" not in x:
            return x
        splits = str(x).split("-")
        return "-".join([splits[0].strip(), splits[-1].strip()])
    
    df["layers_patched"] = df["layers_patched"].map(rename)
    return df

def _label(x):
    if "rename_types" in x:
        return "Rename Types"
    elif "rename_vars" in x:
        return "Rename Variables"
    elif "delete" in x:
        return "Remove Type Annotations"
    else:
        return "NO LABEL"
    
def split_subsets(df):
    df["_kind"] = df["eval_dir"].map(_label)
    df_t = df.loc[df["_kind"] == "Rename Types"]
    df_v = df.loc[df["_kind"] == "Rename Variables"]
    df_d = df.loc[df["_kind"] == "Remove Type Annotations"]
    return df_v, df_t, df_d

def plot_layer_ablation(data: List[pd.DataFrame], outfile, parent_dir = None):
    """
    Figure, 3 subplots (for window size 1,3,5)
    - x: layers patched
    - y: accuracy on ood
    """
    size=14
    fig = plt.figure(figsize=(17,4))
    n_layer = len(data[0]["layers_patched"].unique())
    lengths = [1,3,5]
    ax_legend = None
    for i,df in enumerate(data, start=1):
        subplot = f"1{len(data)}{i}"
        ax = plt.subplot(int(subplot))
        if i == 2:
            ax_legend = ax
        df_v,df_t,df_d = split_subsets(df)
        df_v, df_t, df_d = map(lambda x: layer_sort(x, n_layer, lengths[i-1]), [df_v, df_t, df_d])
        sns.lineplot(data=df_v, x="layers_patched", y="ood_accuracy", markers=True, linewidth=2,color="red")
        sns.lineplot(data=df_t, x="layers_patched", y="ood_accuracy", markers=True, linewidth=2,color="blue")
        sns.lineplot(data=df_d, x="layers_patched", y="ood_accuracy", markers=True, linewidth=2, color="green")
        ax.set_ylim(0,1)
        ax.set_xlim(0,24-lengths[i-1])
        plt.xticks(rotation=45, ha="right")
        ax.set_xlabel("Layers Patched", fontsize=size)
        ax.set_ylabel("Accuracy on Held out set", fontsize=size)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=size)
    
    fig.subplots_adjust(bottom=0.3, wspace=0.3)
    labels = ["Rename Variables", "Rename Types", "Remove Type Annotations"]
    legend_elements = [Line2D([0], [0], color=c, lw=1, label=l) for c,l in zip(["red","blue","green"], labels)]

    ax_legend.legend(handles=legend_elements, bbox_to_anchor=(0.5,-0.3), loc="upper center", fontsize=size, ncol=3)
    plt.savefig(outfile, bbox_inches="tight")
    plt.close()


def plot_finetuning_ablation(df: pd.DataFrame, outfile):
    font=15
    line=3
    size=(6,4)
    plt.figure(figsize=size)
    ax = plt.subplot()
    sns.lineplot(data=df, x="checkpoint", y="humaneval",markers=True,linewidth=line, color="orange")
    plt.xlim(0,5)
    ax.set_xlabel("Epoch number", fontsize=font)
    plt.xticks(fontsize=font)
    plt.yticks(fontsize=font)
    ax.set_ylabel("HumanEval pass@1", fontsize=font)
    plt.tight_layout()
    plt.savefig(outfile.replace(".pdf","_humaneval.pdf"))
    plt.close()
    
    # accuracy fig
    plt.figure(figsize=size)
    ax = plt.subplot()
    sns.lineplot(data=df, x="checkpoint", y="accuracy",markers=True,linewidth=line)
    sns.lineplot(data=df, x="checkpoint", y=[0.9]*len(df["checkpoint"]),markers=True,linewidth=line-1,linestyle="dotted", color="red")
    
    plt.ylim(0,1)
    plt.xlim(0,5)
    plt.xticks(fontsize=font)
    plt.yticks(fontsize=font)
    ax.set_xlabel("Epoch number", fontsize=font)
    ax.set_ylabel("Accuracy", fontsize=font)
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

def plot_box(df: pd.DataFrame, outfile):
    """
    Takes a df from an eval_readme.json[results_per_type] which has: fim_type, count, sum (num_success)
    """
    font_size=15
    # test set distribution
    plt.figure(figsize=(9,5))
    ax = plt.subplot()
    df = df.sort_values("count", ascending=False)
    sx = sns.barplot(x=df["count"], y=df["accuracy"], color="cornflowerblue", errorbar=("pi",50),capsize = 0.1)
    ax.set_ylim(0,1)
    plt.xlabel("Frequency of the type label", fontsize=font_size)
    plt.ylabel("Mean Accuracy", fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    # plt.tight_layout()
    plt.savefig(outfile, bbox_inches="tight")
    plt.close()


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outfile", type=str, required=True)
    parser.add_argument("--results_file", type=str, nargs="+", required=False)
    parser.add_argument("--plotfunc", type=str, required=True, choices=["main", "ftune", "layer","box"])
    parser.add_argument("--parent_dir", type=str, required=False)
    parser.add_argument("--do_transfer", action="store_true", default=False)
    args = parser.parse_args()

    if args.plotfunc == "main":
        data = []
        for f in list(args.results_csv):
            df = pd.read_csv(f)
            data.append(df)
            
        df = pd.concat(data).reset_index()
        # print(df)
        plot_main_results(df, args.outfile, "ood_accuracy", do_transfer=args.do_transfer)
        
    if args.plotfunc == "ftune":
        plot_finetuning_ablation(pd.DataFrame(finetune_starcoder_py_results), args.outfile)
        
    if args.plotfunc == "layer":
        df = pd.concat([pd.read_csv(i) for i in args.results_csv])
        # serapate into 3
        df["lengths"] = df["layers_patched"].map(lambda x: len(str(x).split(",")))
        df0 = df.loc[df["lengths"] == 1]
        df1 = df.loc[df["lengths"] == 3]
        df2 = df.loc[df["lengths"] == 5]
        plot_layer_ablation([df0,df1,df2], args.outfile, args.parent_dir)

    if args.plotfunc == "box":
        data = []
        for res in args.results_file:
            with open(res, "r") as f:
                d = json.load(f)
                data += d["results_per_type"]
        data = pd.DataFrame(data)
        plot_box(data, args.outfile)
