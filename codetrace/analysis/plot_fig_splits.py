from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import argparse
import os
from typing import Optional,Dict,List,Tuple
import textwrap
from tqdm import tqdm
import matplotlib.colors as mcolors
from codetrace.analysis.data import (
    MUTATIONS_RENAMED, 
    ALL_MODELS,
    ResultsLoader,
    ResultKeys
)
from codetrace.analysis.utils import full_language_name, model_n_layer, full_model_name, get_unique_value
from codetrace.analysis.plot_fig_all_models import MODEL_COLORS

SPLIT_NAME = {
    "test": "Steering Vector on Held-out Test Split",
    "steer": "Steering Vector on Steering Split",
    "rand": "Random Steering Vector on Test Split",
}
SPLIT_NAME_WRAPPED = {
    k: textwrap.fill(label, width=20) for k,label in SPLIT_NAME.items()
}

def split_colors(model:str, split:str) -> str:
    model_color = MODEL_COLORS[model]
    # to rgba
    rgba = mcolors.to_rgba(model_color)

    # 3 splits: test, steer, rand
    # test should be darkest and rand lightest of color rgba
    # Determine brightness factor based on the split
    if split == "test":
        factor = 1
    elif split == "steer":
        factor = 0.5
    elif split == "rand":
        factor = 0
    else:
        raise ValueError(f"Invalid split type '{split}'. Must be one of ['test', 'steer', 'rand'].")

    adjusted_rgba = sns.desaturate(rgba, factor)
    # Convert RGBA to a string representation
    return mcolors.to_hex(adjusted_rgba, keep_alpha=True)


def plot_splits(df: pd.DataFrame, outdir: Optional[str] = None):
    # keys
    mutations = sorted(get_unique_value(df, "mutations",7))
    model = get_unique_value(df, "model",1)
    lang = get_unique_value(df, "lang",1)
    interval = get_unique_value(df, "interval",1)

    # axes
    num_cols,num_rows = 4,2
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 7), sharex=True, sharey=True)
    axes = axes.flatten()

    # plot
    for i, mutation in enumerate(mutations):
        subset = df[df["mutations"] == mutation]
        
        for split in ["test","rand","steer"]:
            plot = sns.lineplot(ax=axes[i], data=subset, x="start_layer", y=f"{split}_mean_succ", 
                        label=SPLIT_NAME_WRAPPED[split], color=split_colors(model,split),linewidth=0.8)

        axes[i].set_title(mutation)
        axes[i].set_xlabel("Layer start")
        axes[i].set_ylabel("Accuracy")
        axes[i].set_ylim(0, 1)
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].set_xticks(range(0, model_n_layer(model)-interval, 1))
        axes[i].get_legend().remove()

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    fig.suptitle(f"{full_model_name(model)} {full_language_name(lang)} Steering Performace across Splits", 
                 fontsize=16)
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1, 0.7), fontsize=12)
    plt.xlim(0, model_n_layer(model)-interval)
    if outdir:
        plt.savefig(f"{outdir}/splits-{model}-{lang}.pdf")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=str)
    parser.add_argument("outdir", type=str)
    parser.add_argument("--model",required=True, choices=ALL_MODELS)
    parser.add_argument("--lang", choices=["py","ts"], default="py")
    parser.add_argument("--num-proc", type=int, default=40)
    parser.add_argument("--interval", choices=[1,3,5], type=int, default=5)
    assert os.environ.get('PYTHONHASHSEED',None)=="42",\
        "Set PYTHONHASHSEED to 42 for consistent and reliable caching"
    args = parser.parse_args()
    
    loader = ResultsLoader(Path(args.results_dir).exists(), 
                           cache_dir=args.results_dir)
    keys = ResultKeys(model=args.model,lang=args.lang, interval=args.interval)
    results = loader.load_data(keys)

    SPLITS = ["test","steer","rand"]
    
    processed_results = []
    for r in tqdm(results, "Checking splits"):
        for split in ["test","steer","rand"]:
            assert r[split]
        df = r.to_success_dataframe()
        processed_results.append(df)

    df = pd.concat(processed_results, axis=0).reset_index()
    df_pretty = df.copy()
    df_pretty["mutations"] = df_pretty["mutations"].apply(lambda x: MUTATIONS_RENAMED[x])
    print(df_pretty)
    print(df_pretty.columns)
    os.makedirs(args.outdir, exist_ok=True)
    plot_splits(df_pretty, args.outdir)
