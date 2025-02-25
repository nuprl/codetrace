import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import ast
from codetrace.utils import print_color
from codetrace.analysis.utils import (
    ALL_MODELS,ALL_MUTATIONS, 
    get_unique_value, 
    parse_model_name,
    model_n_layer,
    parse_model_name,
    full_language_name,
    full_model_name
)
from codetrace.analysis.data import ResultKeys,ResultsLoader, SteerResult, cache_to_dir,ANALYSIS_CACHE_DIR
from typing import List,Tuple
import sys
import argparse
from collections import Counter, defaultdict
from tqdm import tqdm
import numpy as np
from scipy.stats import linregress, pearsonr
from codetrace.analysis.plot_fig_all_models import MODEL_COLORS
import seaborn as sns

INITIALS = {
    "types": "T",
    "vars": "V",
    "delete": "R",
    "types_vars": "TV",
    "types_delete": "TR",
    "vars_delete": "VR",
    "delete_vars_types": "A",
}

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

def get_color(model:str, lang:str):
    if lang == "ts":
        return sns.saturate(MODEL_COLORS[model])
    else:
        return sns.desaturate(MODEL_COLORS[model], 0.5)

def get_label(label:str):
    model = parse_model_name(label)
    if "ts" in label:
        return f"{full_language_name('ts')} {full_model_name(model)}"
    else:
        return f"{full_language_name('py')} {full_model_name(model)}"
    
def plot_correlation(df: pd.DataFrame, outfile: str, show_most_common:bool):
    # Assign a unique color to each model-language combination
    unique_combinations = df[["model", "lang"]].drop_duplicates()
    color_map = {f"{row.model}_{row.lang}": get_color(row.model, row.lang)
                for i, row in enumerate(unique_combinations.itertuples(index=False))}
    
    df = df.groupby(["model","mutations","lang"]).agg(
        {"steering_success":"mean","typechecks_before":"mean", "typechecks_after":"mean",
         "prediction_before_steer":list, "prediction_after_steer":list}).reset_index()
    # print(df)
    # Plotting
    plt.figure(figsize=(8, 8))

    x_data = []
    y_data = []
    # Loop through each row of the DataFrame to plot data
    for _, row in df.iterrows():
        model_lang = f"{row.model}_{row.lang}"
        color = color_map[model_lang]
        x_values = row["typechecks_before"]
        y_values = row["steering_success"]
        x_data.append(x_values)
        y_data.append(y_values)
        
        # Scatter plot for the current model-language
        plt.scatter(x_values, y_values, color=color, label=model_lang, alpha=0.7,linewidths=1)
        
        # Annotating mutations
        mutation = row["mutations"]
        plt.annotate(INITIALS[mutation], (x_values, y_values), fontsize=9, color=color,
                xytext=(1, 1), textcoords='offset points')
        
        if show_most_common:
            for presteering in Counter(row["prediction_before_steer"]).most_common(1):
                plt.annotate(presteering[0], (x_values, y_values), fontsize=15, xytext=(1, -12), 
                             textcoords='offset points', color=color)
    
    corr_coefficient, p_value = pearsonr(x_data, y_data)
    print("Correlation", corr_coefficient, p_value)
    slope, intercept, r_value, p_value, std_err = linregress(x_data, y_data)
    line = (slope * np.array(x_data)) + intercept
    # plt.plot(x_data, line, color='black', label=f'Best fit line (r = {corr_coefficient:.2f})', linewidth=0.3)

    handles = []
    seen = set()
    for label,color in color_map.items():
        if label not in seen:
            seen.add(label)
            handles.append((plt.Line2D([0], [0], marker='o', color=color, linestyle='', label=get_label(label)),
                            get_label(label)))

    handles = sorted(handles, key=lambda x:x[1])
    plt.legend(ncols=2, handles=[h[0] for h in handles], loc="lower center")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    # plt.title()
    plt.xlabel("Percent Typechecks before Steer", fontsize=12)
    plt.ylabel("Steering Accuracy", fontsize=12)
    plt.grid(alpha=0.5)
    plt.savefig(outfile)
    plt.grid()



def _load(results_dir: str, interval:int, **kwargs):
    # load model, lang data
    loader = ResultsLoader(Path(results_dir).exists(), cache_dir=results_dir)
    all_results = []
    for model in ALL_MODELS:
        data = loader.load_data(ResultKeys(model=model,interval=interval))
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
    return results

if __name__=="__main__":
    assert os.environ.get('PYTHONHASHSEED',None)=="42",\
        "Set PYTHONHASHSEED to 42 for consistent and reliable caching"
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir")
    parser.add_argument("outdir")
    parser.add_argument("--interval", type=int, choices=[1,3,5], required=True)
    args = parser.parse_args()
    results = _load(**vars(args))
    os.makedirs(args.outdir, exist_ok=True)
    results.to_csv(f"{args.outdir}/correlation-interval_{args.interval}.csv")
    plot_correlation(results, f"{args.outdir}/correlation-interval_{args.interval}.pdf")

