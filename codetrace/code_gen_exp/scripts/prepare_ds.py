import datasets
import sys
import argparse
import glob
import gzip
import json
import pandas as pd

def renamed_ds_to_jsonl():
    """
    Turned unfiltered (renamed) steering ds into MultiPLE jsonl format
    for completion and eval
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--input_ds", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()

    ds = datasets.load_dataset(args.input_ds, split=args.split)

    # create a jsonl file
    ds = ds.rename_columns({"prompt":"old_prompt",
                            "tests": "old_tests",
                            "renamed_tests": "tests",
                            "renamed_program": "prompt", 
                            "results": "old_results"})
    # add an index to name
    ds = ds.map(lambda x,i: {**x, "name": x["name"]+f"_{i}"}, with_indices=True)
    ds = ds.remove_columns(["temperature","top_p","max_tokens"])
    ds.to_json(f"exp_data/{args.output_jsonl}.jsonl", orient="records", lines=True, index=False)

def multiple_results_to_ds():
    """
    Turn a dir of MultiPLE results into a dataset with optional
    filtering for correct/incorrect results
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_ds", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--unfiltered_jsonl", type=str)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--filter", type=str, choices=["correct", "incorrect","none"], required=True)
    
    args = parser.parse_args()
    print("Turning results into a dataset")

    with open(args.unfiltered_jsonl, "r") as f:
        unfiltered_jsonl = f.readlines()
        unfiltered_jsonl = [json.loads(l) for l in unfiltered_jsonl]
        
    def is_correct(data: dict):
        if data is None:
            return None
        n = len(data["results"])
        c = len([True for r in data["results"] if r["status"]
                == "OK" and r["exit_code"] == 0])
        return c/n == 1

    new_ds = []
    for f in glob.glob(args.input_dir + "/*.results.json.gz"):
            
        with gzip.open(f, "rt") as f:
            data = json.load(f)
            correct = is_correct(data)
            original_ex = [e for e in unfiltered_jsonl if e["prompt"] == data["prompt"]]
            assert len(original_ex) == 1
            new_ds.append({
                **data,
                "correct": correct,
                "old_prompt" : original_ex[0]["old_prompt"],
                "renamed_variables" : original_ex[0]["renamed_variables"],
                "renamed_percent" : original_ex[0]["renamed_percent"],
            })
            
    new_ds = pd.DataFrame(new_ds)
    
    if args.filter == "correct":
        new_ds = new_ds[new_ds["correct"]]
    elif args.filter == "incorrect":
        new_ds = new_ds[~new_ds["correct"]]
        
    print(new_ds["correct"].value_counts())

    new_ds = datasets.Dataset.from_pandas(new_ds)
    new_ds = new_ds.rename_columns({"prompt":"renamed_prompt","old_prompt":"original_prompt"})
    new_ds.push_to_hub(args.output_ds)
    
    
if __name__ == "__main__":
    # renamed_ds_to_jsonl()
    multiple_results_to_ds()
    pass