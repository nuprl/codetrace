import datasets
import argparse
import pandas as pd
from codetrace.analysis.data import ResultsLoader,ResultKeys
from tqdm import tqdm
from codetrace.analysis.utils import (
    ALL_MODELS,
    ALL_MUTATIONS,
    full_model_name,
    parse_model_name,
    parse_mutation_name,
    full_language_name,
    MUTATIONS_RENAMED,
    get_unique_value
)
from pathlib import Path
import os
from matplotlib import pyplot as plt
import seaborn as sns
from codetrace.utils import print_color
import itertools as it
import textwrap
from codetrace.analysis.plot_fig_layer_ablations import MUTATION_COLOR

# base_palette = sns.hls_palette(7, h=.5)
basecolor = sorted([(k,v) for k,v in MUTATION_COLOR.items()], key=lambda x:x[0])
basecolor = [x[1] for x in basecolor]
COLORS = list(it.chain(*zip(basecolor, basecolor)))

def compare_icl(steering_df:pd.DataFrame, icl_df:pd.DataFrame, outfile:str) -> pd.DataFrame:
    icl_df["type"] = "icl"
    steering_df["type"] = "steer"
    df = pd.concat([icl_df,steering_df], axis=0)
    df = df[["model","mutations","test_mean_succ","lang","type"]]
    df = df.sort_values(["lang","model","mutations","type"])
    df.to_csv(outfile)
    return df

def plot_icl(df: pd.DataFrame, outfile: str):
    HATCH="////"
    # Plot configuration
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True, sharex=True)
    axes = axes.flatten()

    df["model"] = df["model"].apply(lambda x: textwrap.fill(full_model_name(x), width=15))
    df["type_mut"] = df["type"] + df["mutations"]
    mutations = sorted(get_unique_value(df, "mutations",7))
    
    typ_mut_order = list(it.chain(*[["icl"+mut, "steer"+mut] for mut in mutations]))
    
    # Create bar plots for each available language
    for ax, lang in zip(axes, ["py", "ts"]):
        lang_df = df[df["lang"] == lang]
        barplot = sns.barplot(
            data=lang_df,
            x="model",
            y="test_mean_succ",
            hue="type_mut",
            hue_order=typ_mut_order,
            palette=COLORS,
            ax=ax,
            errorbar=None,
        )

        bars = ax.patches
        """
        order of bars is model, mutation, type, lang
        # eg. 
        # codellama py delete icl
        # starcoder py delete icl
        # codellama py delete steer
        # starcoder py delete steer
        # codellama py types icl etc.
        """
        for i, bar in enumerate(bars):
            if ((i // 5) % 2) == 0 and bar.xy != (0,0):
                bar.set_hatch(HATCH)

        # Modify the legend to reflect hatching
        handles, labels = ax.get_legend_handles_labels()
        for i,(handle, label) in enumerate(zip(handles, labels)):
            mut = parse_mutation_name(label)
            if "icl" in label:
                handle.set_hatch(HATCH)

        ax.legend(handles, labels)
        
        # Set plot details
        ax.set_title(full_language_name(lang), fontsize=15)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_xlabel(None)
        ax.tick_params(axis="x")
        ax.get_legend().remove()

    # Add a global legend
    handles, labels = axes[0].get_legend_handles_labels()
    new_handles = []
    new_labels = []
    seen = set()
    for i,(handle, label) in enumerate(zip(handles, labels)):
        mut = parse_mutation_name(label)
        if mut not in seen and "steer" in label:
            new_handles.append(handle)
            new_labels.append(MUTATIONS_RENAMED[mut])
            seen.add(mut)
        
    new_labels = list(map(lambda x: textwrap.fill(x, width=20),new_labels))
    fig.legend(new_handles, new_labels, bbox_to_anchor=(1, 0.8), fontsize=12)
    plt.suptitle("Comparison of Steering to Prompting with ICL Examples", fontsize=16)
    plt.tight_layout()
    fig.subplots_adjust(right=0.83)
    plt.savefig(outfile)

def _load(results_dir:str, interval:int, icl_dir:str = "results"):
    all_results = []
    loader = ResultsLoader(Path(results_dir).exists(), cache_dir=results_dir)
    for model in tqdm(ALL_MODELS, desc="Loading models data"):
        keys = ResultKeys(model=model, interval=interval)
        results = loader.load_data(keys)
        all_results += results
    
    processed_results = []
    for r in tqdm(all_results, "Loading Test Split Success"):
        assert r.test != None, r.name
        rdf = r.to_success_dataframe("test")
        processed_results.append(rdf)

    # get max results
    df = pd.concat(processed_results, axis=0)
    df = df.groupby(["model","mutations","lang"]).agg({"test_mean_succ":"max"}).reset_index()

    # collect icl df
    all_icl_dfs = []
    for model in ALL_MODELS:
        for muts in ALL_MUTATIONS:
            for lang in ["py","ts"]:
                icl_df = datasets.load_from_disk(f"{icl_dir}/icl_{model}-{lang}-{muts}").to_pandas()
                icl_df["mutations"] = muts
                icl_df["model"] = model
                icl_df["lang"] = lang
                all_icl_dfs.append(icl_df)
            
    df_icl = pd.concat(all_icl_dfs, axis=0).reset_index()
    df_icl = df_icl.groupby(["model","mutations","lang"]).agg({"correct":"mean"}).reset_index()
    df_icl = df_icl.rename(columns={"correct":"test_mean_succ"})
    return df, df_icl

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=str)
    parser.add_argument("outdir", type=str)
    parser.add_argument("--num-proc", type=int, default=40)
    parser.add_argument("--interval", choices=[1,3,5], type=int, default=5)
    assert os.environ.get('PYTHONHASHSEED',None)=="42",\
        "Set PYTHONHASHSEED to 42 for consistent and reliable caching"
    args = parser.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)

    df, df_icl = _load(**vars(args))
    df = compare_icl(df, df_icl, f"{args.outdir}/compare_icl-interval_{args.interval}.csv")
    plot_icl(df, f"{args.outdir}/compare_icl-interval_{args.interval}.pdf")
