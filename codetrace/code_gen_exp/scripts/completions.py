"""
Take completions dir from MultiPLE and turn into a ds
"""
import datasets
import argparse
import glob
import gzip
import json
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--output_ds", type=str, required=True)
parser.add_argument("--input_dir", type=str, required=True)
args = parser.parse_args()

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
        new_ds.append({
            **data,
            "correct": correct,
        })
        

new_ds = pd.DataFrame(new_ds)

# print stats
print(new_ds["correct"].value_counts())

new_ds = datasets.Dataset.from_pandas(new_ds)
new_ds.push_to_hub(args.output_ds)