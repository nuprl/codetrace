# %%
# %config InlineBackend.figure_format = "retina"

import pandas as pd
import json
from pathlib import Path
from matplotlib import pyplot as plt
import seaborn as sns
import datasets
import sys
import os
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
        # df_steering = pd.read_json(steering_path, typ='series').to_frame().T
        df_rand.columns = [f"rand_{c}" for c in df_rand.columns]
        # df_steering.columns = [f"steering_{c}" for c in df_steering.columns]
        df = pd.concat([df, df_rand], axis=1) #df_steering
        df["lang"] = lang
        df["mutations"] = mutations
        df["layers"] = layers
        df["start_layer"] = int(layers.split("_")[0])
        df["model"] = model
        df["num_layers"] = num_layers
        all_dfs.append(df)
    
    return pd.concat(all_dfs), missing_test_results

# %%
def plot_steering_results(df: pd.DataFrame, num_layers: int):
    df = df.reset_index()
    df = df[df["num_layers"] == num_layers]
    mutations = df["mutations"].unique()
    mutations = sorted(mutations)
    num_mutations = len(mutations)
    num_cols = 3
    num_rows = (num_mutations + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 4 * num_rows), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, mutation in enumerate(mutations):
        print(i, mutation)
        subset = df[df["mutations"] == mutation]
        sns.lineplot(ax=axes[i], data=subset, x="start_layer", y="mean_succ", label="Test")
        # sns.lineplot(ax=axes[i], data=subset, x="start_layer", y="steering_mean_succ", label="Steering")
        sns.lineplot(ax=axes[i], data=subset, x="start_layer", y="rand_mean_succ", label="Random")

        axes[i].set_title(mutation)
        axes[i].set_xlabel("Patch layer start")
        axes[i].set_ylabel("Accuracy")
        axes[i].set_ylim(0, 1)  # Set y-axis range from 0 to 1
        axes[i].set_xticks(range(int(df["start_layer"].min()), int(df["start_layer"].max()) + 1, 1))
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(True, which='major', linestyle='-', linewidth=0.5, color='lightgrey')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f"{num_layers} patched layers", fontsize=16)

    plt.tight_layout()
    # plt.show()
    plt.savefig("fig.pdf")

# %%
df, missing_test_results = read_steering_results(sys.argv[1],sys.argv[2])
print(missing_test_results)

# %%
df_pretty = df.copy()
df_pretty["mutations"] = df_pretty["mutations"].apply(lambda x: MUTATIONS_RENAMED[x])
df_pretty = df_pretty.sort_values(["mutations","layers"])
# df_pretty.head(70)

# %%
plot_steering_results(df_pretty, 5)

# %%
# plot_steering_results(df_pretty, 3)

# # %%
# plot_steering_results(df_pretty, 5)

# %%
# datasets.Dataset.from_pandas(df).push_to_hub("nuprl/type-steering", config_name="results")


