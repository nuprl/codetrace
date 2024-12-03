# %%
# %config InlineBackend.figure_format = "retina"

import pandas as pd
import json
from pathlib import Path
import datasets
import sys
import os
from typing import Optional,Dict,List
from tqdm import tqdm
import sys
DIR=os.path.dirname(os.path.abspath(Path(__file__).parent))


# %% [markdown]
# We have several directories named results/steering-LANG-MUTATIONS-LAYERS-MODEL.
# Within each of these directories, there are files called test_results.json.
# Each test_results.json has fields called total and num_succ. Read all of these
# into a pandas dataframe.

# %%
MUTATIONS_RENAMED = {
    "types": "Rename types",
    "vars": "Rename variables",
    "delete": "Remove type annotations",
    "types_vars": "Rename types and variables",
    "types_delete": "Rename types and remove type annotations",
    "vars_delete": "Rename variables and remove type annotations",
    "delete_vars_types": "All edits",
}

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

def batched_process_df_from_hub(batch: List[str], **kwargs) -> List[Dict[str, List]]:
    processed = []
    for item in batch:
        processed.append(process_df_from_hub(item, **kwargs))
    return processed

def read_steering_results_from_hub_multiproc(
    lang:str, model:str, n_layers:int, interval:int, cache_dir: str=None, num_proc: int = None
):
    from codetrace.fast_utils import batched_apply, make_batches
    def _postproc(results):
        dataset, missing = [],[]
        for item in results:
            if isinstance(item["data"], pd.DataFrame):
                dataset.append(item["data"])
            missing += item["missing_results"]
        return dataset, missing
    
    subsets = all_subsets(lang, model, n_layers, interval)
    batches = make_batches(subsets, num_proc)
    results = batched_apply(batches, num_proc, batched_process_df_from_hub, 
                            model=model, cache_dir=cache_dir, verbose=False)
    data, missing_test_results = _postproc(results)
    return pd.concat(data), missing_test_results

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

def read_steering_results(lang:str = "", model:str = ""):
    all_dfs = []
    missing_test_results = []
    for path in Path(f"{DIR}/results").glob(f"steering-{lang}*{model}"):
        test_results_path = path / "test_results.json"
        steering_path = path / "steer_results.json"
        rand_path = path / "test_results_rand.json"
        if not test_results_path.exists() or not rand_path.exists(): #not steering_path.exists() or 
            missing_test_results.append(path)
            continue
        names = path.name.split("-")
        lang, mutations, layers = names[1],names[2],names[3]
        num_layers = len(layers.split("_"))
        df = pd.read_json(test_results_path, typ='series').to_frame().T
        df_rand = pd.read_json(rand_path, typ='series').to_frame().T
        if steering_path.exists():
            df_steering = pd.read_json(steering_path, typ='series').to_frame().T
            df_steering.columns = [f"steering_{c}" for c in df_steering.columns]
        df_rand.columns = [f"rand_{c}" for c in df_rand.columns]
        if steering_path.exists():
            df = pd.concat([df, df_rand, df_steering], axis=1)
        else:
            df = pd.concat([df, df_rand], axis=1)
        df["lang"] = lang
        df["mutations"] = mutations
        df["layers"] = layers
        df["start_layer"] = int(layers.split("_")[0])
        df["model"] = model
        df["num_layers"] = num_layers
        all_dfs.append(df)
    
    return pd.concat(all_dfs), missing_test_results

# %%
def plot_steering_results(df: pd.DataFrame, interval: int, fig_file: Optional[str] = None):
    from matplotlib import pyplot as plt
    import seaborn as sns
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

        sns.lineplot(ax=axes[i], data=subset, x="start_layer", y="mean_succ", label="Test")
        sns.lineplot(ax=axes[i], data=subset, x="start_layer", y="rand_mean_succ", label="Random")
        if "steering_mean_succ" in subset.columns:
            sns.lineplot(ax=axes[i], data=subset, x="start_layer", y="steering_mean_succ", label="Steering")
        
        axes[i].set_title(mutation)
        axes[i].set_xlabel("Patch layer start")
        axes[i].set_ylabel("Accuracy")
        axes[i].set_ylim(0, 1)  # Set y-axis range from 0 to 1
        axes[i].set_xticks(range(int(df["start_layer"].min()), int(df["start_layer"].max()) + 1, 1))
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(True, which='major', linestyle='-', linewidth=0.5, color='lightgrey')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f"{interval} patched layers", fontsize=16)

    plt.tight_layout()

    if fig_file:
        plt.savefig(fig_file)
    else:
        plt.show()

def conditional_prob(var_a: str, var_b: str, df: pd.DataFrame):
    """
    probability of A|B
    
    P(A|B) = P(AnB)/P(B)
    
    """
    prob_a = df[var_a].mean()
    prob_b = df[var_b].mean()
    prob_anb = (df[var_a] & df[var_b]).mean()
    prob_a_given_b = prob_anb / prob_b
    return prob_a, prob_b, prob_anb, prob_a_given_b

def correlation(lang, model):
    all_dfs = []
    for file in Path(f"{DIR}/results").glob(f"steering-{lang}-*-17_18*{model}/test_steering_results_rand"):
        df = datasets.load_from_disk(file).to_pandas()
        all_dfs.append(df)
    df = pd.concat(all_dfs, axis=0).reset_index()
    print(df)
    df["mutated_pred_is_underscore"] = df["mutated_generated_text"] == "__"
    df["success"] = df["steered_predictions"] == df["fim_type"]
    df["mutation_names"] = df["mutation_names"].apply(lambda x: "_".join(x))
    df = df.groupby("mutation_names").agg({"success":"mean", "mutated_pred_is_underscore":"mean"}).reset_index()
    print(df)
    return df.corr("pearson", "success","mutated_pred_is_underscore")

if __name__ == "__main__":
    # %%
    df, missing_test_results = read_steering_results(sys.argv[1],sys.argv[2])
    print(missing_test_results)

    # %%
    df_pretty = df.copy()
    df_pretty["mutations"] = df_pretty["mutations"].apply(lambda x: MUTATIONS_RENAMED[x])
    df_pretty = df_pretty.sort_values(["mutations","layers"])
    # df_pretty.head(70)

    # %%
    # plot_steering_results(df_pretty, 5, "fig.pdf")
    print(correlation(sys.argv[1],sys.argv[2]))
    # %%
    # plot_steering_results(df_pretty, 3)

    # # %%
    # plot_steering_results(df_pretty, 5)

    # %%
    # datasets.Dataset.from_pandas(df).push_to_hub("nuprl/type-steering", config_name="results")


