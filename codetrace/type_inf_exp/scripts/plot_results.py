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

finetune_starcoder_py_results = [
    {"checkpoint":"0","humaneval": 0.14161490683229813, "accuracy":0},
    {"checkpoint":"1","humaneval": 0.04440993788819876, "accuracy":0.86},
    {"checkpoint":"2","humaneval": 0.08167701863354036, "accuracy":0.93},
    {"checkpoint":"3", "humaneval": 0.04534161490683229, "accuracy":0.95},
    {"checkpoint":"4","humaneval": 0.0686335403726708, "accuracy":0.93},
    {"checkpoint":"5","humaneval": 0.049689440993788817, "accuracy":0.91},
    {"checkpoint":"6","humaneval": 0.04751552795031056, "accuracy":0.96},
]

def plot_fientuning_ablation(results: dict, outfile):
    df = pd.DataFrame(results)
    plt.figure(figsize=(12,7))
    ax = plt.subplot()
    sns.lineplot(data=df, x="checkpoint", y="humaneval",markers=True, lw=2)
    sns.lineplot(data=df, x="checkpoint", y="accuracy",markers=True, lw=2)
    plt.ylim(0,1)
    plt.xlim(0,6)
    ax.legend()
    ax.set_xlabel("Checkpoint number", fontsize=12)
    ax.set_ylabel("", fontsize=12)
    custom_handles = [Line2D([0], [0], color='orange', lw=1, label="Accuracy on Held out set"),
                      Line2D([0], [0], color='blue', lw=1, label="Humaneval performance")]
    
    custom_legend = plt.legend(handles=custom_handles, loc='center right', title="Legend")
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

    
# parser = argparse.ArgumentParser()
# parser.add_argument("--outfile", type=str, required=True)
# parser.add_argument("--results_csv", type=str, nargs="+", required=True)
# args = parser.parse_args()
# data = []
# for f in list(args.results_csv):
#     df = pd.read_csv(f)
#     data.append(df)
    
# df = pd.concat(data).reset_index()
# # print(df)
# plot_main_results(df, args.outfile, "ood_accuracy")
plot_fientuning_ablation(finetune_starcoder_py_results, "test_f.pdf")