from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import json
from pathlib import Path
import argparse
import datasets
import sys
import os
from typing import Optional,Dict,List
from tqdm import tqdm
import sys
from codetrace.analysis.data import (
    build_success_data, 
    MUTATIONS_RENAMED, 
    ResultsLoader,
    ResultKeys
)
from codetrace.analysis.utils import full_language_name, model_n_layer

# We have several directories named results/steering-LANG-MUTATIONS-LAYERS-MODEL.
# Within each of these directories, there are files called test_results.json.
# Each test_results.json has fields called total and num_succ. Read all of these
# into a pandas dataframe.

"""
Plotting
"""
def plot_steering_results(
    df: pd.DataFrame, 
    interval: int, 
    fig_file: Optional[str] = None, 
    n_layer: int = None,
    steer_label:str = "Steering",
    test_label:str = "Test",
    rand_label:str = "Random"
):
    labels = {"steer":steer_label, "test":test_label, "rand":rand_label}
    df = df.reset_index()
    df = df[df["num_layers"] == interval]
    mutations = df["mutations"].unique()
    mutations = sorted(mutations)
    num_mutations = len(mutations)
    num_cols = 3
    num_rows = (num_mutations + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 4 * num_rows), sharex=True, sharey=True)
    axes = axes.flatten()
    for i, mutation in enumerate(mutations):
        subset = df[df["mutations"] == mutation]

        for split in ["test","steer","rand"]:
            # Plot line
            if f"{split}_mean_succ" in subset.columns:
                plot = sns.lineplot(ax=axes[i], data=subset, x="start_layer", y=f"{split}_mean_succ", label=labels[split])
                max_test = subset[f"{split}_mean_succ"].max()
                test_color = plot.get_lines()[-1].get_color()  # Extract color from the last line
                axes[i].hlines(y=max_test, xmin=subset["start_layer"].min(), xmax=subset["start_layer"].max(),
                            colors=test_color, linestyles='dashed', label=f"{labels[split]} Max")

        axes[i].set_title(mutation)
        axes[i].set_xlabel("Patch layer start")
        axes[i].set_ylabel("Accuracy")
        axes[i].set_ylim(0, 1)  # Set y-axis range from 0 to 1
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(True, which='major', linestyle='-', linewidth=0.5, color='lightgrey')
        axes[i].legend(loc='best')
        if not n_layer:
            axes[i].set_xticks(range(int(df["start_layer"].min()), int(df["start_layer"].max()) + 1, 1))
        else:
            axes[i].set_xticks(range(0, n_layer-interval+1, 1))

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f"{interval} patched layers", fontsize=16)

    plt.tight_layout()
    if n_layer:
        plt.xlim(0, n_layer-interval+1)
    if fig_file:
        plt.savefig(fig_file)
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["mutations","precomputed","lang_transfer"])
    parser.add_argument("--lang", type=str, 
                                 help="For lang transfer, this is language tested on, not of the steering tensor")
    parser.add_argument("--model", type=str)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--outfile", type=Path)
    parser.add_argument("--interval", type=int, nargs="+", default=[1,3,5])
    parser.add_argument("--num-proc", type=int, default=40)

    assert os.environ.get('PYTHONHASHSEED',None)=="42",\
        "Set PYTHONHASHSEED to 42 for consistent and reliable caching"
    args = parser.parse_args()
    label_kwargs = {}

    keys = ResultKeys(args.model,lang=args.lang)
    loader = ResultsLoader(Path(args.results_dir).exists(), cache_dir=args.results_dir)
    results = loader.load_data(keys)

    if args.command == "mutations":
        df, missing_test_results = build_success_data(results, args.num_proc)
        print(missing_test_results)

        df_pretty = df.copy()
        df_pretty["mutations"] = df_pretty["mutations"].apply(lambda x: MUTATIONS_RENAMED[x])
        print(df_pretty["mutations"].value_counts())
        print(df_pretty.columns)
        df_pretty = df_pretty.sort_values(["mutations","layers"])

    elif args.command == "precomputed":
        df, missing_test_results = build_success_data(results,args.num_proc)
        # no steer for precomputed
        missing_test_results = [m for m in missing_test_results if "/steer" not in m]
        print(missing_test_results)

        df_pretty = df.copy()
        df_pretty["mutations"] = df_pretty["mutations"].apply(lambda x: MUTATIONS_RENAMED[x])
        print(df_pretty["mutations"].value_counts())
        df_pretty = df_pretty.sort_values(["mutations","layers"])

    elif args.command == "lang_transfer":
        # load original language layer sweeps
        df, missing_test_results = build_success_data(results,args.num_proc)
        # keep only rand col, rename test to "steer"
        df_layer_sweep = df[["rand_mean_succ","start_layer","mutations",
                             "layers","num_layers","test_mean_succ"]]
        df_layer_sweep = df_layer_sweep.rename(columns={"test_mean_succ":"steer_mean_succ"}, errors="raise")
        
        # load lang transfer experiment results
        steering_lang = 'py' if args.lang=='ts' else 'ts'
        keys = ResultKeys(model=args.model, lang=args.lang, prefix=f"lang_transfer_{steering_lang}_")
        results = loader.load_data(keys)
        df, missing_test_results = build_success_data(results,args.num_proc)
        df_lang_transfer = df[["test_mean_succ","start_layer","mutations","layers","num_layers"]]

        # join results
        df = pd.merge(df_lang_transfer,df_layer_sweep, 
                      on=["start_layer", "layers","mutations","num_layers"]).reset_index()
        df_pretty = df.copy()
        df_pretty["mutations"] = df_pretty["mutations"].apply(lambda x: MUTATIONS_RENAMED[x])
        print(df_pretty.columns)
        print(df_pretty["mutations"].value_counts())
        df_pretty = df_pretty.sort_values(["mutations","layers"])
        label_kwargs = {"steer_label": f"Original {full_language_name(args.lang)}", "test_label": full_language_name(steering_lang)}
    else:
        raise NotImplementedError("Task not implemented.")
    
    outfile = args.outfile.as_posix()

    for i in args.interval:
        print(f"Plotting interval {i}")
        plot_steering_results(df_pretty, i, outfile.replace(".pdf",f"_{i}.pdf"), 
                              model_n_layer(args.model), **label_kwargs)
