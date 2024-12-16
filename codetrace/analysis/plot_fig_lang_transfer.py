from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import argparse
import os
from typing import Optional,Dict,List,Tuple
from codetrace.analysis.data import (
    MUTATIONS_RENAMED, 
    ALL_MODELS,
    ResultsLoader,
    ResultKeys
)
from codetrace.analysis.utils import full_language_name, model_n_layer, get_unique_value, full_model_name

COLORS = {
    "original": "purple",
    "lang_transfer": "orange"
}
def get_label(kind, tensor_lang, lang):
    if kind=="lang_transfer":
        return full_language_name(tensor_lang) + " Steering Vector"
    else:
        return full_language_name(lang) + " Steering Vector"

def plot_lang_transfer(df: pd.DataFrame, outdir: Optional[str] = None):
    # get values
    mutations =  get_unique_value(df, "mutations",7)
    lang = get_unique_value(df, "lang", 1)
    tensor_lang = "ts" if lang == "py" else "py"
    model = get_unique_value(df, "model", 1)
    interval = get_unique_value(df, "interval", 1)

    # axes plot
    num_cols, num_rows = 4,2
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 7), sharex=True, sharey=True)
    axes = axes.flatten()
    for i, mutation in enumerate(mutations):
        subset = df[df["mutations"] == mutation]
        
        for kind in ["lang_transfer","original"]:
            type_subset = subset[subset["_kind"] == kind]
            plot = sns.lineplot(
                ax=axes[i], 
                data=type_subset, 
                x="start_layer", 
                y="test_mean_succ", 
                label=get_label(kind,tensor_lang,lang),
                color=COLORS[kind])

        axes[i].set_title(MUTATIONS_RENAMED[mutation], fontsize=12)
        axes[i].set_xlabel("Start Layer", fontsize=12)
        axes[i].set_ylabel("Accuracy", fontsize=15)
        axes[i].set_ylim(0, 1)
        axes[i].tick_params(axis='x', labelsize=8, rotation=45)
        axes[i].set_xticks(range(1, model_n_layer(model)-interval+1, 2))
        axes[i].get_legend().remove()

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    _lang = full_language_name(lang)
    _tensor_lang = full_language_name(tensor_lang)
    _model = full_model_name(model)
    fig.suptitle(f"{_model} Steering Performance on {_lang}", fontsize=16)
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.9, 0.7), fontsize=12)
    plt.xlim(0, model_n_layer(model)-interval)
    
    if outdir:
        plt.savefig(f"{outdir}/lang_transfer-{model}-{tensor_lang}_onto_{lang}-interval_{interval}.pdf")
    else:
        plt.show()

# lang is the test lang here
def _load(results_dir:str, model:str, lang:str, interval:int, **kwargs):
    # load original and lang_transfer results
    loader = ResultsLoader(Path(results_dir).exists(), cache_dir=results_dir)
    original_lang_keys = ResultKeys(model=model,lang=lang, interval=interval)
    steer_tensor_lang = "py" if lang == "ts" else "ts"
    lang_transfer_keys = ResultKeys(model=model,lang=lang,interval=interval, 
                        prefix=f"lang_transfer_{steer_tensor_lang}_")
    
    original_results = loader.load_data(original_lang_keys)
    lang_transfer_results = loader.load_data(lang_transfer_keys)

    # check non-null
    for r in original_results:
        assert r["test"], r.name
    for r in lang_transfer_results:
        assert r["test"], r.name
    
    # compute success rate of test
    original_results = [r.to_success_dataframe("test") for r in original_results]
    original_results = pd.concat(original_results, axis=0)
    original_results["_kind"] = "original"

    lang_transfer_results = [r.to_success_dataframe("test") for r in lang_transfer_results]
    lang_transfer_results = pd.concat(lang_transfer_results, axis=0)
    lang_transfer_results["_kind"] = "lang_transfer"

    # join
    df = pd.concat([original_results, lang_transfer_results], axis=0).reset_index()
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=str)
    parser.add_argument("outdir", type=str)
    parser.add_argument("--model",required=True, choices=ALL_MODELS)
    parser.add_argument("--lang", choices=["py","ts"], default="py", 
        help="This is the language tested on, not steered")
    parser.add_argument("--num-proc", type=int, default=40)
    parser.add_argument("--interval", choices=[5], type=int, default=5, 
                        help="Only ran these experiments for 5")
    assert os.environ.get('PYTHONHASHSEED',None)=="42",\
        "Set PYTHONHASHSEED to 42 for consistent and reliable caching"
    args = parser.parse_args()
    df = _load(**vars(args))
    os.makedirs(args.outdir, exist_ok=True)
    plot_lang_transfer(df, args.outdir)
