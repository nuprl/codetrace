import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sys
import os
import json
from glob import glob
import platform
import nvsmi

def _plot_results(results : pd.DataFrame, window_size, outfile: str, layer_n=24) -> None:
    """
    Plot confidence interval
    - x axis: layer
    - y axis: accuracy
    """
    results = results[["patched_layer", "correct_steer"]]
    plt.figsize=(10, 10)
    # group by layer and type
    grouped = results.groupby(["patched_layer"]).agg({"correct_steer": "mean"}).reset_index()
    # set to str
    grouped["patched_layer"] = grouped["patched_layer"].astype(str)
    
    def _prettify(x):
        return "-".join([str(i) for i in x])
    
    all_layers = list(range(layer_n))
    zipped = list(zip([all_layers[i:i+window_size] for i in all_layers]))
    windows = [j[0] for j in zipped if len(j[0]) == window_size]
    windows = [_prettify(i) for i in windows]
    print(grouped["patched_layer"])
    
    # add column with sort_idx
    grouped["sort_idx"] = grouped["patched_layer"].apply(lambda x: windows.index(x))
    # sort grouped by window key
    grouped = grouped.sort_values(by="sort_idx")
    # set sort_idx as new index
    grouped.set_index("sort_idx", inplace=True)
    print(grouped)
    
    # plot accuracy per layer
    fig, ax = plt.subplots()
    y = []
    x_original = []
    for i in range(len(grouped)):
        y.append(grouped.iloc[i]["correct_steer"])
        x_original.append(grouped.iloc[i]["patched_layer"])

    # sort 
    if len(y) < layer_n:
        layer_n = len(y)
    x = range(layer_n)

    ax.plot(x, y)
    # set x ticks limit to 0-max layer
    ax.set_xlim(0, layer_n-1)
    ax.set_xticks(list(range(layer_n)))
    
    ax.set_xticklabels(x_original)
    plt.xticks(rotation=45, ha="right")

    # draw vertical gridlines
    ax.grid()
    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy")
    plt.xticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()

def _plot_only(window_size, outdir, resultsdir, layersdir):
    try:
        nvsmi.get_gpus()
        import datasets
        if os.path.exists(f"{outdir}/{resultsdir}"):
            results = datasets.load_from_disk(f"{outdir}/{resultsdir}")
            _plot_results(results.to_pandas(),window_size, f"{outdir}/{resultsdir}.pdf")
    except:
        if os.path.exists(f"{outdir}/{layersdir}"):
            results = []
            for layer_res in glob(f"{outdir}/{layersdir}/layer_*.csv"):
                results.append(pd.read_csv(layer_res))
            results = pd.concat(results)
            _plot_results(results, window_size, f"{outdir}/{resultsdir}.pdf")
        else:
            print(f"No results found in {outdir}/{resultsdir} or {outdir}/{layersdir}")

def main(args):
    args.fim_placeholder = False
    args.custom_decoder = False
    args.patch_mode = "add"
    args.steering_outfile = None
    
    if args.plot_only:
        # _plot_only(args.sliding_window_size, args.outdir, "ablation_results", "layer_results")
        _plot_only(args.sliding_window_size, args.outdir, "ood_ablation_results", "ood_layer_results")
        return
    return

if __name__ == "__main__":
    if sys.argv[1].endswith(".json") and "args" in sys.argv[1]:
        with open(sys.argv[1], "r") as f:
            args = json.load(f)
        args = argparse.Namespace(**args)
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_name", type=str, required=True)
        parser.add_argument("--device", type=str, default="cuda")
        parser.add_argument("--dataset", type=str, required=True)
        parser.add_argument("--tokens_to_patch", type=str, required=True)
        parser.add_argument("--batch_size", type=int, default=10)
        parser.add_argument("--out_dir", type=str)
        parser.add_argument("--plot-only", action="store_true")
        args = parser.parse_args()

    main(args)
