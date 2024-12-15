from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import argparse
import textwrap
import os
from typing import Optional,Dict,List
from tqdm import tqdm
from codetrace.analysis.data import (
    MUTATIONS_RENAMED, 
    ALL_MODELS,
    ResultsLoader,
    ResultKeys
)
from codetrace.analysis.utils import full_language_name, model_n_layer, get_unique_value, full_model_name
import itertools as it

basecolor = sns.hls_palette(7, l=0.7, h=0.1)
MUTATION_COLOR = {
    "delete": basecolor[0],
    "types": basecolor[1],
    "vars": basecolor[2],
    "vars_delete": basecolor[3],
    "types_delete": basecolor[4],
    "types_vars": basecolor[5],
    "delete_vars_types": basecolor[6]
}
# MUTATION_COLOR = {
#     "delete": "mediumorchid",
#     "types": "darkgreen",
#     "vars": "coral",
#     "vars_delete": "deepskyblue",
#     "types_delete": "maroon",
#     "types_vars": "goldenrod",
#     "delete_vars_types": "royalblue"
# }

"""
Plotting
"""
WRAPPED_MUTATIONS_RENAMED = {
    k: textwrap.fill(label, width=20) for k,label in MUTATIONS_RENAMED.items()
}

def plot_all_layers(df: pd.DataFrame, outdir: Optional[str] = None):
    # get labels
    # mutations = sorted(["types","delete","vars"])
    mutations = sorted(get_unique_value(df, "mutations",7))
    lang = get_unique_value(df, "lang", 1)
    model = get_unique_value(df, "model", 1)

    # axes setup
    num_cols,num_rows=3,1
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 4), sharex=False, sharey=True)
    axes = axes.flatten()

    for i,interval in enumerate([1,3,5]):
        subset_interval = df[df["interval"] == interval]
        for mutation in mutations:
            subset = subset_interval[subset_interval["mutations"] == mutation]
            plot = sns.lineplot(
                ax=axes[i],
                data=subset,
                x="start_layer",
                y=f"test_mean_succ", 
                label=WRAPPED_MUTATIONS_RENAMED[mutation],
                color=MUTATION_COLOR[mutation])
            max_test = subset[f"test_mean_succ"].max()
            axes[i].hlines(
                y=max_test,
                xmin=subset["start_layer"].min(),
                xmax=subset["start_layer"].max(),
                colors=MUTATION_COLOR[mutation],
                linestyles='dashed',
                linewidth=0.7,
                linestyle=(0, (5, 10)))
            
        axes[i].set_title(f"Steering {interval} Layers", fontsize=12)
        axes[i].set_xlabel("Start Layer", fontsize=10)
        axes[i].set_ylabel("Accuracy", fontsize=12)
        axes[i].set_ylim(0, 1)  # Set y-axis range from 0 to 1
        axes[i].get_legend().remove()

        axes[i].tick_params(axis='x', rotation=45)
        axes[i].set_xlim(0, model_n_layer(model)+1-interval)
        axes[i].set_xticks(range(0, model_n_layer(model)+1-interval, 2))

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(
        f"{full_model_name(model)} Performance on {full_language_name(lang)} after Steering at Varying Intervals", 
        fontsize=16)

    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1, 1.1), fontsize=12)
    fig.subplots_adjust(right=0.8)
    if outdir:
        plt.savefig(f"{outdir}/intervals-{model}-{lang}.pdf")
    else:
        plt.show()

def _load(results_dir:str, model:str, lang:str, **kwargs):
    # load results
    loader = ResultsLoader(Path(results_dir).exists(), cache_dir=results_dir)
    keys = ResultKeys(model=model,lang=lang)
    results = loader.load_data(keys)
    
    processed_results = []
    for r in tqdm(results, "checking"):
        assert r.test != None, r.name
        rdf = r.to_success_dataframe("test")
        processed_results.append(rdf)

    df = pd.concat(processed_results, axis=0)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=str)
    parser.add_argument("outdir", type=str)
    parser.add_argument("--model", required=True, choices=ALL_MODELS)
    parser.add_argument("--lang", choices=["py","ts"], default="py")
    parser.add_argument("--num-proc", type=int, default=40)
    assert os.environ.get('PYTHONHASHSEED',None)=="42",\
        "Set PYTHONHASHSEED to 42 for consistent and reliable caching"
    args = parser.parse_args()
    
    df = _load(**vars(args))
    os.makedirs(args.outdir, exist_ok=True)
    plot_all_layers(df, args.outdir)
