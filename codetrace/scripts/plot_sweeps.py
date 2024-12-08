from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import json
from pathlib import Path
import argparse
import datasets
import sys
import os
from typing import Optional,Dict,List
from tqdm import tqdm
import sys
from codetrace.fast_utils import batched_apply, make_batches

# We have several directories named results/steering-LANG-MUTATIONS-LAYERS-MODEL.
# Within each of these directories, there are files called test_results.json.
# Each test_results.json has fields called total and num_succ. Read all of these
# into a pandas dataframe.

MUTATIONS_RENAMED = {
    "types": "Rename types",
    "vars": "Rename variables",
    "delete": "Remove type annotations",
    "types_vars": "Rename types and variables",
    "types_delete": "Rename types and remove type annotations",
    "vars_delete": "Rename variables and remove type annotations",
    "delete_vars_types": "All edits",
}

"""
Utils code
"""

def fmt_language(lang: str) ->str:
    if lang == "py":
        return "Python"
    elif lang == "ts":
        return "TypeScript"
    else:
        raise ValueError(f"Not found {lang}")

def get_ranges(num_layers: int, interval: int):
    for i in range(0, num_layers):
        if i + interval <= num_layers:
            yield "_".join(map(str, range(i, i + interval)))

def all_subsets(lang:str, model:str, n_layers:int, interval:int):
    subsets = []
    ranges = get_ranges(n_layers, interval)
    for r in ranges:
        for m in MUTATIONS_RENAMED.keys():
            subsets.append(f"steering-{lang}-{m}-{r}-{model}")
    return subsets

def model_n_layer(model: str) -> int:
    if "qwen" in model.lower():
        return 28
    elif "codellama" in model.lower():
        return 32
    elif "starcoderbase-1b" in model.lower():
        return 24
    elif "starcoderbase-7b" in model.lower():
        return 42
    else:
        print(f"Model {model} model_n_layer not implemented!")
        return None

"""
Loading code
"""
def _format_results(df: pd.DataFrame) -> pd.DataFrame:
    df["succ"] = df["steered_predictions"] == df["fim_type"]
    return pd.DataFrame.from_records([
        {"num_succ":df["succ"].sum(), 
         "tot_succ":df["succ"].count(),
         "mean_succ":df["succ"].mean(),}])

def process_df_from_hub(subset:str, model:str, cache_dir:str, verbose:bool=False) -> Dict[str, List]:
    missing_test_results = []
    try:
        ds = datasets.load_dataset("nuprl-staging/type-steering-results", name=subset, 
                                    cache_dir=cache_dir)
    except Exception as e:
        if verbose:
            print(e)
        missing_test_results.append(subset)
        return {"missing_results": missing_test_results, "data": None}

    test_results = ds["test"]
    rand_results = ds["rand"]
    steering_results = ds.get("steer", None)
    if not steering_results:
        missing_test_results.append(subset + "/steer")
    names = subset.split("-")
    lang, mutations, layers = names[1],names[2],names[3]
    num_layers = len(layers.split("_"))
    df = test_results.to_pandas()
    df = _format_results(df)
    df_rand = rand_results.to_pandas()
    df_rand = _format_results(df_rand)
    df_rand.columns = [f"rand_{c}" for c in df_rand.columns]
    if steering_results:
        df_steering = steering_results.to_pandas()
        df_steering = _format_results(df_steering)
        df_steering.columns = [f"steering_{c}" for c in df_steering.columns]
    if steering_results:
        df = pd.concat([df, df_rand, df_steering], axis=1)
    else:
        df = pd.concat([df, df_rand], axis=1)
    df["lang"] = lang
    df["mutations"] = mutations
    df["layers"] = layers
    df["start_layer"] = int(layers.split("_")[0])
    df["model"] = model
    df["num_layers"] = num_layers
    return {"data": df, "missing_results": missing_test_results}

def process_df_local(path: Path, model:str, ignore_rand:bool = False) ->  Dict[str, List]:
    missing_test_results = []
    test_results_path = path / "test_results.json"
    steering_path = path / "steer_results.json"
    rand_path = path / "test_results_rand.json"
    if not test_results_path.exists():
        missing_test_results.append(test_results_path)
        return {"data": None, "missing_results": missing_test_results }
    if not ignore_rand and not rand_path.exists():
        missing_test_results.append(rand_path)
        return {"data": None, "missing_results": missing_test_results}
    
    names = path.name.split("-")
    lang, mutations, layers = names[1],names[2],names[3]
    num_layers = len(layers.split("_"))
    df = pd.read_json(test_results_path, typ='series').to_frame().T
    
    all_data = [df]

    if steering_path.exists():
        df_steering = pd.read_json(steering_path, typ='series').to_frame().T
        df_steering.columns = [f"steering_{c}" for c in df_steering.columns]
        all_data.append(df_steering)
    else:
        missing_test_results.append(steering_path)

    if not ignore_rand:
        df_rand = pd.read_json(rand_path, typ='series').to_frame().T
        df_rand.columns = [f"rand_{c}" for c in df_rand.columns]
        all_data.append(df_rand)
    
    df = pd.concat(all_data, axis=1)
    df["lang"] = lang
    df["mutations"] = mutations
    df["layers"] = layers
    df["start_layer"] = int(layers.split("_")[0])
    df["model"] = model
    df["num_layers"] = num_layers
    return {"data": df, "missing_results": missing_test_results}

"""
List processing
"""
def read_steering_results_from_hub(
    lang:str, model:str, n_layers:int, interval:int, cache_dir: str=None
):
    all_dfs = []
    missing_test_results = []
    for subset in tqdm(all_subsets(lang, model, n_layers, interval), "fetching subsets"):
        output = process_df_from_hub(subset, model, cache_dir)
        if isinstance(output["data"], pd.DataFrame):
            all_dfs.append(output["data"])
        missing_test_results += output["missing_results"]
    
    return pd.concat(all_dfs), missing_test_results

def read_steering_results(results_dir: str, lang:str = "", model:str = ""):
    all_dfs = []
    missing_test_results = []
    for path in tqdm(list(Path(results_dir).glob(f"steering-{lang}*{model}")), desc="Reading"):
        output = process_df_local(path)
        if isinstance(output["data"], pd.DataFrame):
            all_dfs.append(output["data"])
        missing_test_results += output["missing_results"]
    
    return pd.concat(all_dfs), missing_test_results

"""
Multiproc code
"""
def _postproc(results):
    dataset, missing = [],[]
    for item in results:
        if isinstance(item["data"], pd.DataFrame):
            dataset.append(item["data"])
        missing += item["missing_results"]
    return dataset, missing


def batched_process(batch: List[str], fn: callable, **kwargs) -> List[Dict[str, List]]:
    processed = []
    for item in batch:
        processed.append(fn(item, **kwargs))
    return processed

def read_steering_results_from_hub_multiproc(
    lang:str, model:str, n_layers:int, interval:int, cache_dir: str=None, num_proc: int = None
):
    subsets = all_subsets(lang, model, n_layers, interval)
    batches = make_batches(subsets, num_proc)
    results = batched_apply(batches, num_proc, batched_process, fn=process_df_from_hub, 
                            model=model, cache_dir=cache_dir, verbose=False)
    data, missing_test_results = _postproc(results)
    return pd.concat(data), missing_test_results

def read_steering_results_multiproc(
    results_dir: str, lang:str, model:str, num_proc: int = None
):
    subsets = list(Path(results_dir).glob(f"steering-{lang}*{model}"))
    batches = make_batches(subsets, num_proc)
    results = batched_apply(batches, num_proc, batched_process, fn=process_df_local, model=model)
    data, missing_test_results = _postproc(results)
    return pd.concat(data), missing_test_results

def read_steering_results_precomputed_multiproc(
    results_dir: str, lang:str, model:str, num_proc: int = None, prefix: str = "precomputed_"
):
    subsets = list(Path(results_dir).glob(f"{prefix}steering-{lang}-*-{model}"))
    batches = make_batches(subsets, num_proc)
    results = batched_apply(batches, num_proc, batched_process, 
                            fn=process_df_local, model=model, 
                            ignore_rand=("lang_transfer" in prefix))
    data, missing_test_results = _postproc(results)
    return pd.concat(data), missing_test_results
"""
Plotting
"""
def plot_steering_results(
    df: pd.DataFrame, 
    interval: int, 
    fig_file: Optional[str] = None, 
    n_layer: int = None,
    steer_label:str = "Steering",
    test_label:str = "Test",
    rand_label:str = "Random"
):
    df = df.reset_index()
    df = df[df["num_layers"] == interval]
    mutations = df["mutations"].unique()
    mutations = sorted(mutations)
    num_mutations = len(mutations)
    num_cols = 3
    num_rows = (num_mutations + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 4 * num_rows), sharex=True, sharey=True)
    axes = axes.flatten()
    for i, mutation in enumerate(mutations):
        subset = df[df["mutations"] == mutation]

                # Plot Test line
        test_plot = sns.lineplot(ax=axes[i], data=subset, x="start_layer", y="mean_succ", label=test_label)
        max_test = subset["mean_succ"].max()
        test_color = test_plot.get_lines()[-1].get_color()  # Extract color from the last line
        axes[i].hlines(y=max_test, xmin=subset["start_layer"].min(), xmax=subset["start_layer"].max(),
                       colors=test_color, linestyles='dashed', label=f"{test_label} Max")

        # Plot Steering line if the column exists
        if "steering_mean_succ" in subset.columns:
            steering_plot = sns.lineplot(ax=axes[i], data=subset, x="start_layer", y="steering_mean_succ", 
                                         label=steer_label)
            max_steering = subset["steering_mean_succ"].max()
            steering_color = steering_plot.get_lines()[-1].get_color()  # Extract color
            axes[i].hlines(y=max_steering, xmin=subset["start_layer"].min(), xmax=subset["start_layer"].max(),
                           colors=steering_color, linestyles='dashed', label=f"{steer_label} Max")

        # Plot Random line
        random_plot = sns.lineplot(ax=axes[i], data=subset, x="start_layer", y="rand_mean_succ", label=rand_label)
        max_random = subset["rand_mean_succ"].max()
        random_color = random_plot.get_lines()[-1].get_color()  # Extract color
        axes[i].hlines(y=max_random, xmin=subset["start_layer"].min(), xmax=subset["start_layer"].max(),
                       colors=random_color, linestyles='dashed', label=f"{rand_label} Max")

        axes[i].set_title(mutation)
        axes[i].set_xlabel("Patch layer start")
        axes[i].set_ylabel("Accuracy")
        axes[i].set_ylim(0, 1)  # Set y-axis range from 0 to 1
        if not n_layer:
            axes[i].set_xticks(range(int(df["start_layer"].min()), int(df["start_layer"].max()) + 1, 1))
        else:
            axes[i].set_xticks(range(0, n_layer-interval+1, 1))
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(True, which='major', linestyle='-', linewidth=0.5, color='lightgrey')
        axes[i].legend(loc='best')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f"{interval} patched layers", fontsize=16)

    plt.tight_layout()
    if n_layer:
        plt.xlim(0, n_layer-interval+1)
    if fig_file:
        plt.savefig(fig_file)
    else:
        plt.show()

"""
Metrics
"""
def conditional_prob(var_a: str, var_b: str, df: pd.DataFrame):
    """
    probability of A|B
    
    P(A|B) = P(AnB)/P(B)
    
    """
    prob_a = df[var_a].mean()
    prob_b = df[var_b].mean()
    prob_anb = (df[var_a] & df[var_b]).mean()
    prob_a_given_b = 0 if prob_anb == 0 else prob_anb / prob_b
    return prob_a, prob_b, prob_anb, prob_a_given_b


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    parser_muts = subparsers.add_parser("mutations")
    parser_muts.add_argument("--lang", type=str)
    parser_muts.add_argument("--model", type=str)
    parser_muts.add_argument("--results-dir", type=Path)
    parser_muts.add_argument("--outfile", type=Path)
    parser_muts.add_argument("--interval", type=int, nargs="+", default=[1,3,5])

    parser_precomputed = subparsers.add_parser("precomputed")
    parser_precomputed.add_argument("--lang", type=str)
    parser_precomputed.add_argument("--model", type=str)
    parser_precomputed.add_argument("--results-dir", type=Path)
    parser_precomputed.add_argument("--outfile", type=Path)
    parser_precomputed.add_argument("--interval", type=int, nargs="+", default=[1,3,5])

    parser_transfer = subparsers.add_parser("lang_transfer")
    parser_transfer.add_argument("--lang", type=str, help="Language tested on, not of the steering tensor")
    parser_transfer.add_argument("--model", type=str)
    parser_transfer.add_argument("--results-dir", type=Path)
    parser_transfer.add_argument("--outfile", type=Path)
    parser_transfer.add_argument("--interval", type=int, nargs="+", default=[1,3,5])

    args = parser.parse_args()
    label_kwargs = {}

    if args.command == "mutations":
        df, missing_test_results = read_steering_results_multiproc(args.results_dir,args.lang,args.model,40)
        print(missing_test_results)

        df_pretty = df.copy()
        df_pretty["mutations"] = df_pretty["mutations"].apply(lambda x: MUTATIONS_RENAMED[x])
        print(df_pretty["mutations"].value_counts())
        df_pretty = df_pretty.sort_values(["mutations","layers"])

    elif args.command == "precomputed":
        df, missing_test_results = read_steering_results_precomputed_multiproc(
            args.results_dir,
            args.lang,
            args.model,
            40
        )
        # no steer for precomputed
        missing_test_results = [m for m in missing_test_results if "steer_results.json" not in m.as_posix()]
        print(missing_test_results)

        df_pretty = df.copy()
        df_pretty["mutations"] = df_pretty["mutations"].apply(lambda x: MUTATIONS_RENAMED[x])
        print(df_pretty["mutations"].value_counts())
        df_pretty = df_pretty.sort_values(["mutations","layers"])

    elif args.command == "lang_transfer":
        df, missing_test_results = read_steering_results_multiproc(args.results_dir,args.lang,args.model,40)
        # keep only rand col
        df_layer_sweep = df[["rand_mean_succ","start_layer","mutations",
                             "layers","num_layers","mean_succ"]]
        df_layer_sweep = df_layer_sweep.rename(columns={"mean_succ":"steering_mean_succ"}, errors="raise")
        
        steering_lang = 'py' if args.lang=='ts' else 'ts'
        df, missing_test_results = read_steering_results_precomputed_multiproc(
            args.results_dir,
            args.lang,
            args.model,
            40,
            prefix=f"lang_transfer_{steering_lang}_"
        )
        df_lang_transfer = df[["mean_succ","start_layer","mutations","layers","num_layers"]]
        # no steer for precomputed
        df = pd.merge(df_lang_transfer,df_layer_sweep, 
                      on=["start_layer", "layers","mutations","num_layers"]).reset_index()
        df_pretty = df.copy()
        df_pretty["mutations"] = df_pretty["mutations"].apply(lambda x: MUTATIONS_RENAMED[x])
        print(df_pretty["mutations"].value_counts())
        df_pretty = df_pretty.sort_values(["mutations","layers"])
        label_kwargs = {"steer_label": f"Original {fmt_language(args.lang)}", "test_label": fmt_language(steering_lang)}
    else:
        raise NotImplementedError("Task not implemented.")
    
    outfile = args.outfile.as_posix()

    for i in args.interval:
        print(f"Plotting interval {i}")
        plot_steering_results(df_pretty, i, outfile.replace(".pdf",f"_{i}.pdf"), 
                              model_n_layer(args.model), **label_kwargs)
