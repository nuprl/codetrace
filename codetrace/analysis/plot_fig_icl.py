import datasets
import argparse
import pandas as pd
from codetrace.analysis.data import ResultsLoader,ResultKeys
from tqdm import tqdm
from codetrace.analysis.utils import ALL_MODELS,ALL_MUTATIONS,MUTATIONS_RENAMED
from pathlib import Path
import os
from matplotlib import pyplot as plt
import seaborn as sns
import itertools as it

COLORS=[
    (0.1216, 0.4667, 0.7059, 1.0),  # Opaque Blue
    (0.1216, 0.4667, 0.7059, 0.2),  # Transparent Blue
    (1.0, 0.4980, 0.0549, 1.0),     # Opaque Orange
    (1.0, 0.4980, 0.0549, 0.2),     # Transparent Orange
    (0.1725, 0.6275, 0.1725, 1.0),  # Opaque Green
    (0.1725, 0.6275, 0.1725, 0.2),  # Transparent Green
    (0.8392, 0.1529, 0.1569, 1.0),  # Opaque Red
    (0.8392, 0.1529, 0.1569, 0.2),  # Transparent Red
    (0.5804, 0.4039, 0.7412, 1.0),  # Opaque Purple
    (0.5804, 0.4039, 0.7412, 0.2),  # Transparent Purple
    (0.5490, 0.3373, 0.2941, 1.0),  # Opaque Brown
    (0.5490, 0.3373, 0.2941, 0.2),  # Transparent Brown
    (0.8902, 0.4667, 0.7608, 1.0),  # Opaque Pink
    (0.8902, 0.4667, 0.7608, 0.2)   # Transparent Pink
]
COLORS.reverse()

base_palette = sns.hls_palette(7, h=.5)
COLORS = [
    sns.desaturate(color,0.2) if i % 2 == 0 else sns.saturate(color)  # Alternate transparent/opaque
    for i, color in enumerate(it.chain(*zip(base_palette, base_palette[::-1])))
]

def compare_icl(steering_df:pd.DataFrame, icl_df:pd.DataFrame, outfile:str) -> pd.DataFrame:
    icl_df["type"] = "icl"
    steering_df["type"] = "steer"
    df = pd.concat([icl_df,steering_df], axis=0)
    df = df[["model","mutations","test_mean_succ","lang","type"]]
    df = df.sort_values(["lang","model","mutations","type"])
    df.to_csv(outfile)
    return df

def plot_icl(df:pd.DataFrame, outfile:str):
    # Ensure that we only process available languages
    available_langs = df['lang'].unique()

    # Plot configuration
    fig, axes = plt.subplots(1, len(available_langs), figsize=(15, 6), sharey=True)

    # If only one subplot, wrap it in a list
    if len(available_langs) == 1:
        axes = [axes]

    df["type_mut"] = df["type"] + df["mutations"]
    # Create bar plots for each available language
    for ax, lang in zip(axes, available_langs):
        lang_df = df[df['lang'] == lang]
        barplot = sns.barplot(
            data=lang_df,
            x="model",
            y="test_mean_succ",
            hue="type_mut",
            palette=sns.color_palette(COLORS),
            ax=ax,
            errorbar=None,
            dodge=True
        )
            
        ax.get_legend().remove()
        ax.set_title(f"Performance for {lang.upper()}")
        ax.set_ylabel("Mean Success Rate")
        ax.set_xlabel("Model")
        ax.tick_params(axis='x', rotation=45)

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, bbox_to_anchor=(1, 1), title="Type/Mutation")
    
    plt.tight_layout()
    # plt.show()
    fig.subplots_adjust(right=0.85)
    plt.savefig(outfile)


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

    all_results = []
    loader = ResultsLoader(Path(args.results_dir).exists(), cache_dir=args.results_dir)
    for model in tqdm(ALL_MODELS, desc="models"):
        keys = ResultKeys(model=model, interval=args.interval)
        results = loader.load_data(keys)
        all_results += results
    
    processed_results = []
    for r in tqdm(all_results, "checking"):
        assert r.test != None, r.name
        rdf = r.to_success_dataframe("test")
        processed_results.append(rdf)

    df = pd.concat(processed_results, axis=0)
    df = df.groupby(["model","mutations","lang"]).agg({"test_mean_succ":"max"}).reset_index()

    # collect icl df
    all_icl_dfs = []
    for model in ALL_MODELS:
        for muts in ALL_MUTATIONS:
            for lang in ["py","ts"]:
                icl_df = datasets.load_from_disk(f"results/icl_{model}-{lang}-{muts}").to_pandas()
                icl_df["mutations"] = muts
                icl_df["model"] = model
                icl_df["lang"] = lang
                all_icl_dfs.append(icl_df)
            
    df_icl = pd.concat(all_icl_dfs, axis=0).reset_index()
    df_icl = df_icl.groupby(["model","mutations","lang"]).agg({"correct":"mean"}).reset_index()
    df_icl = df_icl.rename(columns={"correct":"test_mean_succ"})

    df = compare_icl(df, df_icl, f"{args.outdir}/compare_icl-interval_{args.interval}.csv")
    plot_icl(df, f"{args.outdir}/compare_icl-interval_{args.interval}.pdf")
