import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import ast
from codetrace.analysis.utils import ALL_MODELS,ALL_MUTATIONS
from codetrace.analysis.data import ResultKeys,ResultsLoader
import sys
import argparse

def plot_correlation(df: pd.DataFrame, outfile: str):
    # Assign a unique color to each model-language combination
    unique_combinations = df[["model", "lang"]].drop_duplicates()
    color_map = {f"{row.model}_{row.lang}": plt.cm.tab10(i) 
                for i, row in enumerate(unique_combinations.itertuples(index=False))}
    df["presteering"] = df["change"].apply(lambda x: ast.literal_eval(x)[0])
    df = df.groupby(["mutation","lang","start_layer"]).agg(
        {"steering_success":"mean","typechecks_before":"mean", "presteering":"unique"}).reset_index()
    df = df.groupby(["mutation","lang","typechecks_before","presteering"]).agg(
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
        for i,presteering in enumerate(row["presteering"]):
            plt.annotate(presteering, (x_values[i], y_values[i]), fontsize=9, xytext=(1, -1), textcoords='offset points')

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    handles = [plt.Line2D([0], [0], marker='o', color=color, linestyle='', label=label) 
            for label, color in color_map.items()]
    plt.legend(handles=handles, title="Model_Language", loc="best")

    # Customizing the plot
    plt.xlabel("Probability typechecks_before")
    plt.ylabel("Probability steering_success")
    plt.grid(alpha=0.5)
    plt.savefig(outfile)
    plt.legend()
    plt.grid()

if __name__=="__main__":
    assert os.environ.get('PYTHONHASHSEED',None)=="42",\
        "Set PYTHONHASHSEED to 42 for consistent and reliable caching"
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir")
    parser.add_argument("outfile")
    args = parser.parse_args()
    
    all_results = []
    for model in ALL_MODELS:
        loader = ResultsLoader(Path(args.results_dir).exists(), cache_dir=args.results_dir)
        data = loader.load_data(ResultKeys(model=model,interval=5))
        all_results += data
    results = pd.concat([r.to_errors_dataframe("test") for r in all_results], axis=0)
    plot_correlation(results, args.outfile)

