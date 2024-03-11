"""
Code for fig 1.

Given sets of matching correct-incorrect prompts, patch the <fim_middle> token to make the incorrect prompts correct.
Show accuracy at every layer, for each type. 200 examples per type.

TODO:
- how to do most efficiently?
    - process by type
    - get all patches for a type
    - apply all patches for a type
"""
import pandas as pd
import torch
import json
import pickle
from nnsight import LanguageModel
import datasets
from codetrace.utils import placeholder_to_std_fmt, STARCODER_FIM
from matplotlib import pyplot as plt
from typing import List, Tuple, Union
from collections import Counter
from codetrace.interp_utils import collect_hidden_states_at_tokens, insert_patch
from tqdm import tqdm

def _batched_predictions(
    model : LanguageModel,
    prompts_pos : List[str],
    prompts_neg : List[str],
    target_type : List[str],
    tokens_to_patch : Union[List[str],List[int],str,int],
    batch_size : int,
    outfile : str,
) -> pd.DataFrame:
    """
    Get patched predictions for a given type
    """
    batch_pos = [prompts_pos[i:i+batch_size] for i in range(0, len(prompts_pos), batch_size)]
    batch_neg = [prompts_neg[i:i+batch_size] for i in range(0, len(prompts_neg), batch_size)]
    batch_target = [target_type[i:i+batch_size] for i in range(0, len(target_type), batch_size)]
    layers_to_patch = list(range(len(model.transformer.h)))
    
    results = []
    for i in tqdm(range(len(batch_pos)), desc="Batch predictions"):
        # get patches for this type
        patches = collect_hidden_states_at_tokens(model, batch_pos[i], tokens_to_patch)

        for layer in layers_to_patch:
            res : TraceResult = insert_patch(model, 
                                            batch_neg[i], 
                                            patches, 
                                            layer, 
                                            tokens_to_patch, 
                                            patch_mode = "subst", 
                                            collect_hidden_states=False)
            prompt_len = len(batch_neg[i])
            logits : LogitResult = res.decode_logits(prompt_idx=list(range(prompt_len)), do_log_probs=False)

            for j in range(prompt_len):
                tok = logits[-1][j].tokens(model.tokenizer)
                tok = tok[0].strip()
                results.append({
                    "prompt_pos": batch_pos[i][j],
                    "prompt_neg": batch_neg[i][j],
                    "target_type": batch_target[i][j],
                    "prediction": tok,
                    "correct": (1 if tok.startswith(batch_target[i][j]) else 0),
                    "patched_token": tokens_to_patch,
                    "patched_layer": layer
                })
                
        # save results
        pd.DataFrame(results).to_csv(outfile)
                
    return pd.DataFrame(results)


def _process_renamed_by_type(dataset : datasets.Dataset, tokenizer, num_per_type=200) -> Tuple[List[str], List[str], List[str]]:
    """
    Process by type
    """
    if "<FILL>" in dataset[0]["fim_program"]:
        dataset = dataset.map(lambda x: {"fim_program": placeholder_to_std_fmt(x["fim_program"], STARCODER_FIM),
                                         "renamed_fim_program": placeholder_to_std_fmt(x["renamed_fim_program"], STARCODER_FIM)})
    # truncate to one token
    dataset = dataset.map(lambda x: {"fim_type": tokenizer.decode(tokenizer.encode(x["fim_type"])[0])})
    
    # if any type has less than num_per_type, remove types
    counts = Counter(dataset["fim_type"])
    types = [t for t in list(counts.keys()) if counts[t] >= num_per_type]
    count_types = {t:0 for t in types}
    
    # get examples per type
    positive_prompts = []
    negative_prompts = []
    target_types = []
    for ex in dataset:
        if ex["fim_type"] not in types:
            continue
        elif count_types[ex["fim_type"]] < num_per_type:
            positive_prompts.append(ex["fim_program"])
            negative_prompts.append(ex["renamed_fim_program"])
            target_types.append(ex["fim_type"])
            count_types[ex["fim_type"]] += 1
    print(f"Counts: {count_types}")
    return positive_prompts, negative_prompts, target_types


def _plot_results(results : pd.DataFrame, outfile: str) -> None:
    """
    Plot confidence interval
    - x axis: layer
    - y axis: accuracy
    """
    # group by layer and type
    grouped = results.groupby(["patched_layer", "target_type"]).agg({"correct": "mean"}).reset_index()
    # plot different color lines for each type
    fig, ax = plt.subplots()
    for t in grouped["target_type"].unique():
        subset = grouped[grouped["target_type"] == t]
        ax.plot(subset["patched_layer"], subset["correct"], label=t)
    ax.legend()
    # set x ticks limit to 0-max layer
    ax.set_xlim(0, max(grouped["patched_layer"]))
    ax.set_xticks(list(range(max(grouped["patched_layer"])+1)))
    # draw vertical gridlines
    ax.grid(axis="x")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy")
    plt.xticks(fontsize=8)
    plt.savefig(outfile)
    plt.close()
    

def main(args):
    if args.plot_only:
        results = pd.read_csv(args.outfile)
        _plot_results(results, args.outfile.replace(".csv", ".pdf"))
        return
    
    model = LanguageModel(args.model_name, device_map=args.device)
    dataset = datasets.load_dataset(args.dataset, split="train")
    positive_prompts, negative_prompts, target_types = _process_renamed_by_type(dataset, model.tokenizer, args.num_per_type)
    print(f"Num prompts: {len(positive_prompts)}")
    predictions = _batched_predictions(model, 
                                       positive_prompts, 
                                       negative_prompts, 
                                       target_types, 
                                       args.tokens_to_patch, 
                                       args.batch_size, 
                                       args.outfile)
    # save as csv
    predictions.to_csv(args.outfile, index=False)
    _plot_results(predictions, args.outfile.replace(".csv", ".pdf"))
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--num_per_type", type=int, default=200)
    parser.add_argument("--tokens_to_patch", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--outfile", type=str, default="predictions.csv")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    print(args)
    main(args)