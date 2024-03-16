"""
Code for fig 1.

Given sets of correct-incorrect prompts, compute steering vector from the averages of FIM.
Then, compute accuracy per layer patch onto incorrect prompts.
"""
import pandas as pd
import torch
import json
import os
from nnsight import LanguageModel
import datasets
from codetrace.utils import placeholder_to_std_fmt, STARCODER_FIM
from matplotlib import pyplot as plt
from typing import List, Tuple, Union
from collections import Counter
from codetrace.interp_utils import collect_hidden_states_at_tokens, insert_patch
from tqdm import tqdm
import argparse
import sys
import numpy as np
from glob import glob
from codetrace.type_inf_exp.scripts.steering import keep_columns, _get_steering_tensor, steer, _get_ood

def steering_ablation(model, steering_tensor,negative_prompts,args, is_ood):
    if is_ood:
        ood_flag = "ood_"
    else:
        ood_flag = ""
        
    os.makedirs(f"{args.expdir}/{ood_flag}layer_results", exist_ok=True)
    results = []
    
    all_layers = list(range(steering_tensor.shape[0]))
    # create sliding window of layers
    zipped = list(zip([all_layers[i:i+args.sliding_window_size] for i in all_layers]))
    batch_layers = [j[0] for j in zipped if len(j[0]) == args.sliding_window_size]
    
    for layer in batch_layers:
        layername = "-".join([str(i) for i in layer])
        args.steering_outfile = f"{args.expdir}/{ood_flag}layer_results/layer_{layername}.json"
        args.layers_to_patch = layer
        layer_res_ds = steer(model, negative_prompts, steering_tensor, args)
        # add a layer column
        layer_res_ds = layer_res_ds.map(lambda x : {"patched_layer" : layername})
        # save layer results
        layer_res_ds.to_csv(args.steering_outfile.replace(".json", ".csv"))
        results.append(layer_res_ds)
    results = datasets.concatenate_datasets(results)
    return results

def _process_prompts(dataset : datasets.Dataset, tokenizer) -> Tuple[List[str]]:
    """
    Process by type
    """
    # reprocess correct to be sure
    if "generated_text" in dataset.column_names:
        # recompute "correct" sanity check: generated text starts with first token of fim_type
        dataset = dataset.map(lambda x : {"correct" : x["generated_text"].strip().startswith(
            tokenizer.decode(tokenizer.encode(x["fim_type"])[0])), **x})
    
    if "renamed_fim_program" in dataset.column_names:
        if "<FILL>" in dataset[0]["fim_program"]:
            dataset = dataset.map(lambda x: {"fim_program": placeholder_to_std_fmt(x["fim_program"], STARCODER_FIM),
                                            "renamed_fim_program": placeholder_to_std_fmt(x["renamed_fim_program"], STARCODER_FIM)})
        positive = keep_columns(dataset, ["fim_program", "fim_type","hexsha"])
        negative = keep_columns(dataset, ["renamed_fim_program", "fim_type","hexsha"]).rename_column("renamed_fim_program", "fim_program")
        return positive, negative
    else:
        if "<FILL>" in dataset[0]["fim_program"]:
            dataset = dataset.map(lambda x: {"fim_program": placeholder_to_std_fmt(x["fim_program"], STARCODER_FIM)})
        positive = dataset.filter(lambda x: x["correct"])
        negative = dataset.filter(lambda x: not x["correct"])
        return positive, negative


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
    if os.path.exists(f"{outdir}/{resultsdir}"):
        results = datasets.load_from_disk(f"{outdir}/{resultsdir}")
        _plot_results(results.to_pandas(),window_size, f"{outdir}/{resultsdir}.pdf")
    elif os.path.exists(f"{outdir}/{layersdir}"):
        results = []
        for layer_res in glob(f"{outdir}/{layersdir}/layer_*.csv"):
            results.append(pd.read_csv(layer_res))
        results = pd.concat(results)
        _plot_results(results, window_size, f"{outdir}/{resultsdir}.pdf")
    else:
        print(f"No results found in {outdir}/{resultsdir} or {outdir}/{layersdir}")

def main(args):
    raise NotImplementedError("This script has not been updated!")
    args.fim_placeholder = False
    args.custom_decoder = False
    args.patch_mode = "add"
    args.steering_outfile = None
    
    if args.plot_only:
        _plot_only(args.sliding_window_size, args.expdir, "ablation_results", "layer_results")
        _plot_only(args.sliding_window_size, args.expdir, "ood_ablation_results", "ood_layer_results")
        return
    
    expdir = f"{args.expdir}/{args.run_id}"
    os.makedirs(expdir, exist_ok=True)
    
    # save args
    with open(f"{expdir}/args.json", "w") as f:
        save_args = args
        save_args.plot_only = True
        json.dump(vars(save_args), f, indent=4)
        
    model = LanguageModel(args.model_name, device_map=args.device)
    dataset = datasets.load_dataset(args.dataset, split="train")
    if args.shuffle:
        dataset = dataset.shuffle(seed=42)
    
    positive_prompts, negative_prompts = _process_prompts(dataset, model.tokenizer)
    positive_prompts, negative_prompts, negative_ood = _get_ood(positive_prompts, negative_prompts, args)
    
    if args.max_size > -1:
        positive_prompts = positive_prompts.select(range(args.max_size))
        negative_prompts = negative_prompts.select(range(args.max_size))
        negative_ood = negative_ood.select(range(args.max_size))
        
    data_info = f"""
    Positive prompts: {len(positive_prompts)}
    Negative prompts: {len(negative_prompts)}
    Negative OOD prompts: {len(negative_ood)}
    """
    print(data_info)
    with open(f"{args.outdir}/data_readme.md","w") as f:
        f.write(data_info)
    
    steering_tensor = _get_steering_tensor(model, positive_prompts, negative_prompts, args)
    
    # steering_ablation
    if os.path.exists(f"{args.outdir}/ablation_results"):
        results = datasets.load_from_disk(f"{args.outdir}/ablation_results")
    else:
        results = steering_ablation(model, steering_tensor, negative_prompts, args, False)
        results.save_to_disk(f"{args.outdir}/ablation_results")
    
    # plot
    _plot_results(results.to_pandas(), args.sliding_window_size, f"{args.outdir}/ablation_results.pdf")
    
    # OOD steering_ablation
    if os.path.exists(f"{args.outdir}/ood_ablation_results"):
        results_ood = datasets.load_from_disk(f"{args.outdir}/ood_ablation_results")
    else:
        results_ood = steering_ablation(model, steering_tensor, negative_ood, args, True)
        results_ood.save_to_disk(f"{args.outdir}/ood_ablation_results")
    
    # plot
    _plot_results( results_ood.to_pandas(), args.sliding_window_size, f"{args.outdir}/ood_ablation_results.pdf")


if __name__ == "__main__":
    if sys.argv[1].endswith(".json") and "args" in sys.argv[1]:
        with open(sys.argv[1], "r") as f:
            args = json.load(f)
        args = argparse.Namespace(**args)
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_name", type=str, required=True)
        parser.add_argument("--device", type=str, default="cuda")
        parser.add_argument("--source-dataset", type=str, required=True)
        parser.add_argument("--tokens_to_patch", type=str, required=True)
        parser.add_argument("--batch_size", type=int, default=10)
        parser.add_argument("--datadir", type=str)
        parser.add_argument("--expdir", type=str)
        parser.add_argument("--plot-only", action="store_true")
        args = parser.parse_args()

    main(args)
