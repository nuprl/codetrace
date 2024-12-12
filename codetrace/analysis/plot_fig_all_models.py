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
    ALL_MODELS,
    ALL_MUTATIONS,
    ResultsLoader,
    ResultKeys
)
from codetrace.analysis.utils import full_language_name, model_n_layer

MODEL_COLORS = {
    "starcoderbase-7b": "purple",
    "starcoderbase-1b": "cyan",
    "CodeLlama-7b-Instruct-hf": "orange",
    "Llama-3.2-3B-Instruct": "green",
    "qwen2p5_coder_7b_base": "red"
}
"""
Plotting
"""
def plot_all_models(
    df: pd.DataFrame, 
    fig_file: Optional[str] = None,
    interval: int = 5
):
    df = df.reset_index()
    print(df.columns)
    df["model_num_layers"] = df["model"].apply(lambda x: model_n_layer(x) - interval)
    df["relative_num_layers"] = df["start_layer"] / df["model_num_layers"]
    mutations = df["mutations"].unique()
    mutations = sorted(mutations)
    num_cols = 4
    num_rows = 2

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 7), sharex=True, sharey=True)
    axes = axes.flatten()
    for i, mutation in enumerate(mutations):
        subset = df[df["mutations"] == mutation]
        for model in ALL_MODELS:
            model_subset = subset[subset["model"] == model]
            plot = sns.lineplot(ax=axes[i], data=model_subset, x="relative_num_layers", y=f"test_mean_succ", 
                                label=model,color=MODEL_COLORS[model],linewidth=0.2)
            max_test = model_subset[f"test_mean_succ"].max()
            test_color = MODEL_COLORS[model]  # Extract color from the last line
            axes[i].hlines(y=max_test, xmin=model_subset["relative_num_layers"].min(), xmax=model_subset["relative_num_layers"].max(),
                        colors=test_color, linestyles='dashed',linewidth=0.5,
                        linestyle=(0, (5, 10)))

        axes[i].set_title(mutation)
        axes[i].set_xlabel("Relative layer start")
        axes[i].set_ylabel("Accuracy")
        axes[i].set_ylim(0, 1)  # Set y-axis range from 0 to 1
        axes[i].tick_params(axis='x', rotation=45)
        # axes[i].grid(True, which='major', linestyle='-', linewidth=0.3, color='lightgrey')
        # axes[i].legend(loc='best')
        axes[i].set_xticks([i/10 for i in range(0,11,5)])
        axes[i].get_legend().remove()
    
    # empty_ax = axes[-1]
    # empty_ax.axis('off')  # Turn off the axis
    # handles, labels = axes[0].get_legend_handles_labels()
    # empty_ax.legend(handles, labels, loc='center')
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f"Model Steering Performace on Test", fontsize=16)

    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.9, 0.7), fontsize=12)
    plt.xlim(0, 1)
    if fig_file:
        plt.savefig(fig_file)
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=str)
    parser.add_argument("outfile", type=str)
    parser.add_argument("--lang", choices=["py","ts"], default="py")
    parser.add_argument("--num-proc", type=int, default=40)

    assert os.environ.get('PYTHONHASHSEED',None)=="42",\
        "Set PYTHONHASHSEED to 42 for consistent and reliable caching"
    args = parser.parse_args()
    
    all_results = []
    loader = ResultsLoader(Path(args.results_dir).exists(), 
                           cache_dir=args.results_dir)
    for model in tqdm(ALL_MODELS, desc="models"):
        keys = ResultKeys(model=model,lang=args.lang, interval=5)
        results = loader.load_data(keys)
        all_results += results
    
    processed_results = []
    for r in tqdm(all_results, "checking"):
        assert r.test != None, r.name
        rdf = r.to_dataframe("test")
        rdf = rdf.groupby("mutations").agg(
            {"test_is_success":"mean",
             "model":"unique",
             "start_layer":"unique"}).reset_index()
        processed_results.append(rdf)

    df = pd.concat(processed_results, axis=0)
    df = df.rename(columns={"test_is_success":"test_mean_succ"})
    for key in ["model","start_layer"]:
        df[key] = df[key].apply(lambda x: x[0] if len(x) == 1 else 1/0)
    df_pretty = df.copy()
    df_pretty["mutations"] = df_pretty["mutations"].apply(lambda x: MUTATIONS_RENAMED[x])
    print(df_pretty)
    print(df_pretty.columns)
    plot_all_models(df_pretty, args.outfile)
