from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import argparse
import os
from typing import Optional,Dict,List
from tqdm import tqdm
from codetrace.analysis.data import (
    MUTATIONS_RENAMED, 
    ALL_MODELS,
    ResultsLoader,
    ResultKeys
)
from codetrace.analysis.utils import (
    full_language_name,
    full_model_name,
    model_n_layer,
    get_unique_value
)

MODEL_COLORS = {
    "starcoderbase-7b": "violet",
    "starcoderbase-1b": "cornflowerblue",
    "CodeLlama-7b-Instruct-hf": "orange",
    "Llama-3.2-3B-Instruct": "green",
    "qwen2p5_coder_7b_base": "red"
}
"""
Plotting
"""
def plot_all_models(df: pd.DataFrame, outdir: Optional[str] = None):
    # preprocess
    interval = get_unique_value(df, "interval", 1)
    df["model_num_layers"] = df["model"].apply(lambda x: model_n_layer(x) - interval)
    df["relative_num_layers"] = df["start_layer"] / df["model_num_layers"]
    mutations = get_unique_value(df, "mutations", 7)
    mutations = sorted(mutations)
    lang = get_unique_value(df, "lang", 1)

    # axes setup
    num_cols, num_rows = 4, 2
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 7), sharex=True, sharey=True)
    axes = axes.flatten()

    # plot
    for i, mutation in enumerate(mutations):
        subset = df[df["mutations"] == mutation]
        for model in ALL_MODELS:
            model_subset = subset[subset["model"] == model]
            sns.lineplot(
                ax=axes[i],
                data=model_subset,
                x="relative_num_layers",
                y=f"test_mean_succ", 
                label=full_model_name(model),
                color=MODEL_COLORS[model])
            max_test = model_subset["test_mean_succ"].max()
            axes[i].hlines(
                y=max_test, 
                xmin=model_subset["relative_num_layers"].min(), 
                xmax=model_subset["relative_num_layers"].max(),
                colors=MODEL_COLORS[model],
                linestyles='dashed',
                linewidth=0.7,
                linestyle=(0, (5, 10)))

        axes[i].set_title(MUTATIONS_RENAMED[mutation])
        axes[i].set_xlabel("Relative Start Layer",fontsize=12)
        axes[i].set_ylabel("Accuracy",fontsize=12)
        axes[i].set_ylim(0, 1)  # Set y-axis range from 0 to 1
        axes[i].tick_params(axis='x')
        axes[i].set_xticks([0,0.5,1])
        axes[i].get_legend().remove()
    
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f"Steering Performance {full_language_name(lang)}", fontsize=15)

    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.1, 0.8), fontsize=15)
    plt.xlim(0, 1)
    if outdir:
        plt.savefig(f"{outdir}/all_models-lang_{lang}-interval_{interval}.pdf")
    else:
        plt.show()

def _load(results_dir:str,lang:str,interval:int):
    all_results = []
    loader = ResultsLoader(Path(results_dir).exists(), cache_dir=results_dir)
    for model in tqdm(ALL_MODELS, desc="models"):
        keys = ResultKeys(model=model,lang=lang, interval=interval)
        results = loader.load_data(keys)
        all_results += results
    
    processed_results = []
    for r in tqdm(all_results, "Loading Test split Success Rate"):
        assert r.test != None, r.name
        rdf = r.to_success_dataframe("test")
        processed_results.append(rdf)

    df = pd.concat(processed_results, axis=0)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=str)
    parser.add_argument("outdir", type=str)
    parser.add_argument("--lang", choices=["py","ts"], default="py")
    parser.add_argument("--num-proc", type=int, default=40)
    parser.add_argument("--interval", choices=[1,3,5], type=int, default=5)
    assert os.environ.get('PYTHONHASHSEED',None)=="42",\
        "Set PYTHONHASHSEED to 42 for consistent and reliable caching"
    args = parser.parse_args()
    
    df = _load(**vars(args))
    os.makedirs(args.outdir, exist_ok=True)
    plot_all_models(df, args.outdir)
