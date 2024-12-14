import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import ast
from codetrace.utils import print_color
from codetrace.analysis.utils import ALL_MODELS,ALL_MUTATIONS, get_unique_value, parse_model_name
from codetrace.analysis.data import ResultKeys,ResultsLoader, SteerResult, cache_to_dir,ANALYSIS_CACHE_DIR
from typing import List,Tuple
import sys
import argparse
from collections import Counter, defaultdict
from tqdm import tqdm

def _is_correct_max(results: List[SteerResult]) -> bool:
    """
    Results is a list of steer results for all the max steering
    performance. Check that a given (lang, mut, model) triple only appears once.
    Or if it does appear more than once, check all have the same max value.
    """
    triples = []
    for r in results:
        triples.append((parse_model_name(r.name), r.lang, r.mutations))
    if len(triples) == len(set(triples)):
        return True

    max_values = defaultdict(set)
    counter = Counter(triples)
    for r in results:
        if counter[(parse_model_name(r.name), r.lang, r.mutations)] > 1:
            max_values[(parse_model_name(r.name), r.lang, r.mutations)].add(
                get_unique_value(
                    r.to_success_dataframe("test"), # cached so not expensive,
                    "test_mean_succ",
                    1
                )
            )
    return all([len(v)== 1 for v in max_values.values()])


def plot_correlation(df: pd.DataFrame, outfile: str):
    # Assign a unique color to each model-language combination
    unique_combinations = df[["model", "lang"]].drop_duplicates()
    color_map = {f"{row.model}_{row.lang}": plt.cm.tab10(i) 
                for i, row in enumerate(unique_combinations.itertuples(index=False))}
    
    df = df.groupby(["mutations","lang","start_layer"]).agg(
        {"steering_success":"mean","typechecks_before":"mean", "prediction_before_steer":"unique"}).reset_index()
    print(df)
    df = df.groupby(["mutations","lang","typechecks_before","prediction_before_steer"]).agg(
        {"steering_success": "max"}).reset_index()
    # Plotting
    plt.figure(figsize=(12, 8))

    # Loop through each row of the DataFrame to plot data
    for _, row in df.iterrows():
        model_lang = f"{row['model']}_{row['lang']}"
        color = color_map[model_lang]
        x_values = row["typechecks_before"]
        y_values = row["steering_success"]
        
        # Scatter plot for the current model-language
        plt.scatter(x_values, y_values, color=color, label=model_lang, alpha=0.7)
        
        # Annotating mutations
        for i, mutation in enumerate(row["mutations"]):
            plt.annotate(mutation, (x_values[i], y_values[i]), fontsize=9, xytext=(1, 1), textcoords='offset points')
        for i,presteering in enumerate(row["prediction_before_steer"]):
            plt.annotate(presteering, (x_values[i], y_values[i]), fontsize=9, xytext=(1, -1), textcoords='offset points')

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    handles = [plt.Line2D([0], [0], marker='o', color=color, linestyle='', label=label) 
            for label, color in color_map.items()]
    plt.legend(handles=handles, title="Model_Language", loc="best")

    # Customizing the plot
    plt.xlabel("Probability Typechecks before Steer")
    plt.ylabel("Steering Accuracy")
    plt.grid(alpha=0.5)
    plt.savefig(outfile)
    plt.legend()
    plt.grid()

if __name__=="__main__":
    assert os.environ.get('PYTHONHASHSEED',None)=="42",\
        "Set PYTHONHASHSEED to 42 for consistent and reliable caching"
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir")
    parser.add_argument("outdir")
    parser.add_argument("--interval", type=int, choices=[1,3,5], required=True)
    args = parser.parse_args()
    
    # load model, lang data
    loader = ResultsLoader(Path(args.results_dir).exists(), cache_dir=args.results_dir)
    all_results = []
    for model in ALL_MODELS:
        data = loader.load_data(ResultKeys(model=model,interval=args.interval))
        all_results += data
    
    # find most successful layer for lang,mut,model
    # we do this so we don't have to typecheck all data, only typecheck most successful

    max_df = pd.concat([r.to_success_dataframe("test") for r in all_results], axis=0)
    max_df = max_df.groupby(["model","mutations","lang"]).agg({"test_mean_succ":"max"}).reset_index()
    print(max_df)
    preproc_results = []
    for r in all_results:
        # cached so not expensive
        df = r.to_success_dataframe("test")
        assert len(df) == 1
        keys = df.to_dict(orient="records")[0]
        
        # if df is in the most successful results, add
        max_succ = max_df[
            (max_df["model"] == keys["model"]) & \
            (max_df["mutations"] == keys["mutations"]) & \
            (max_df["lang"] == keys["lang"]) & \
            (max_df["test_mean_succ"] == keys["test_mean_succ"])
        ]
        
        if len(max_succ) == 1:
            preproc_results.append(r)
        elif len(max_succ) > 1:
            raise ValueError(f"Should not happen: {max_succ}")

    assert _is_correct_max(preproc_results)
    preproc_results = list(set(preproc_results))
    print(len(preproc_results))

    # typecheck
    results = []
    for r in tqdm(preproc_results, desc="Typechecking"):
        results.append(r.to_errors_dataframe("test"))
    
    results = pd.concat(results, axis=0)
    os.makedirs(args.outdir, exist_ok=True)
    plot_correlation(results, f"{args.outdir}/correlation-interval_{args.interval}.pdf")

