import datasets
import argparse
from collections import Counter
import pandas as pd
from tqdm import tqdm
from copy import deepcopy

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

def get_ranges(num_layers: int, interval: int):
    for i in range(0, num_layers):
        if i + interval <= num_layers:
            yield ",".join(map(str, range(i, i + interval)))
    
def analysis(train_ds: datasets.Dataset, test_ds:datasets.Dataset):
    """
    This tests if fixing the source activation helps steering towards correct dir.
    RQs:
        - if a specific type is disproportionately represented in train set MGTs,
            are corresponding MGTs in test set the ones that get correctly steered?
        - in these cases, check distribution of gold FIM types to rule out steering tensor
            only lears a specific transformation t_a->t_b

    Solution:
    
    Augment test_ds with a column percent_train_mgt, which measures how often in % the MGT of
    an item appears in corresponding train_ds (join by mutations).

    Group by layers. Group by mutated generated text (MGT). Group by mutation.
    Compute correlaation of SUCC and PERCENT_TRAIN_MGT
    """
    test_df = test_ds.to_pandas()
    train_df = train_ds.to_pandas()
    test_df["success"] = test_df["steered_predictions"] == test_df["fim_type"]

    # compute test mutations to mgt for each fim_type
    muts_to_mgt_counter = {}
    train_df["mutation_names"] = train_df["mutation_names"].apply(lambda x: ",".join(x))
    test_df["mutation_names"] = test_df["mutation_names"].apply(lambda x: ",".join(x))
    
    train_df = train_df.groupby("mutation_names").agg({
        "mutated_generated_text": lambda x: dict(Counter(x)),
        "mutation_names":"first"
    })
    for _index, row in train_df.iterrows():
        assert row["mutation_names"] not in muts_to_mgt_counter.keys()
        muts_to_mgt_counter[row["mutation_names"]] = row["mutated_generated_text"]

    def get_percent_mgt(row):
        mutation_dict = muts_to_mgt_counter.get(row["mutation_names"], {})
        numerator = mutation_dict.get(row["mutated_generated_text"], 0)
        denominator = sum(mutation_dict.values())
        return numerator / denominator if denominator > 0 else 0
    
    test_df["percent_train_mgt"] = test_df.apply(get_percent_mgt, axis=1)
    test_df[["percent_train_mgt","layers","mutation_names",
             "mutated_generated_text","steered_predictions",
             "fim_type","success"]].to_csv("test_df.csv")
    
    test_df_grouped = test_df.groupby(["mutated_generated_text","layers"]).agg(
        {"success":"mean", "percent_train_mgt":"mean", "fim_type": lambda x: dict(Counter(x)),
         "mutation_names":"unique"}
    ).reset_index().sort_values(["percent_train_mgt","success","layers"])
    test_df_grouped.to_csv("test_df_grouped.csv")
    prob_a, prob_b, prob_anb, prob_a_given_b = conditional_prob("success","percent_train_mgt", test_df)
    print(prob_a, prob_b, prob_anb, prob_a_given_b)
    # prob_a, prob_b, prob_anb, prob_a_given_b = conditional_prob("success","percent_train_mgt", test_df_grouped)
    # print(prob_a, prob_b, prob_anb, prob_a_given_b)
    # print(test_df.corr("pearson","success", "percent_train_mgt"))

ALL_MUTS =[
    "types",
    "delete",
    "vars",
    "types,delete",
    "vars,delete",
    "types,vars",
    "delete,vars,types"
]
def main_from_hub(model: str, lang: str, num_layers: int, interval: int):
    train_ds = []
    test_ds = []
    all_layers = list(get_ranges(num_layers, interval))
    for mutations in ALL_MUTS:
        for layers in tqdm(all_layers, desc="Collecting all layer steering data"):
            mutation_underscored = mutations.replace(",", "_")
            layers_underscored = layers.replace(",", "_")
            try:
                test_item = datasets.load_from_disk(
                    f"results/steering-{lang}-{mutation_underscored}-{layers_underscored}-{model}/test_steering_results"
                )
                train_item = datasets.load_from_disk(
                    f"results/steering-{lang}-{mutation_underscored}-{layers_underscored}-{model}/steer_steering_results"
                )
            except Exception as e:
                print(e)
                continue
            test_ds.append(test_item.add_column("layers", [layers_underscored]*len(test_item)))
            train_ds.append(train_item.add_column("layers", [layers_underscored]*len(train_item)))
    train_ds = datasets.concatenate_datasets(train_ds)
    test_ds = datasets.concatenate_datasets(test_ds)
    print(train_ds, test_ds)
    analysis(train_ds, deepcopy(train_ds))

def main_from_disk(dirname: str):
    test_ds = datasets.load_from_disk(f"{dirname}/test_steering_results")
    train_ds = datasets.load_from_disk(f"{dirname}/steer_steering_results")
    print(train_ds, test_ds)
    analysis(train_ds, deepcopy(train_ds))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    parser_from_hub = subparsers.add_parser("hub")
    parser_from_hub.add_argument("--model", type=str, required=True)
    parser_from_hub.add_argument("--lang", type=str, required=True)
    parser_from_hub.add_argument("--num-layers", type=int, required=True)
    parser_from_hub.add_argument("--interval", type=int, required=True)

    parser_from_disk = subparsers.add_parser("disk")
    parser_from_disk.add_argument("dirname", type=str)

    args = parser.parse_args()

    if args.command == "hub":
        main_from_hub(**vars(args))
    elif args.command == "disk":
        main_from_disk(args.dirname)
    else:
        raise ValueError(f"Unknown command: {args.command}")