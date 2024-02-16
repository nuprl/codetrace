import datasets
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import *
import json
import pandas as pd

completions_file = sys.argv[1]

with open(completions_file, "r") as f:
    completions = json.load(f)
    
df = pd.DataFrame(completions)

def label(row):
    prompt = row["prompt"]
    solution = row["solution"].strip()
    completion = row["generated"].strip()
    if completion == solution:
        return "correct"
    elif completion.startswith(solution) and completion in prompt:
        return "overfull-correct"
    elif len(completion) > len(solution) and completion.startswith(solution):
        return "overfull-partial-credit-hallucinate"
    else:
        return "incorrect"
    
# add a "correctness" column with correct, incorrect, or maybe
df["correctness"] = df.apply(lambda row: label(row), axis=1)

print(df["correctness"].value_counts())
# print total
print(f"Total: {len(df)}")
# print accuracy count
print(f"Correct: {len(df[df['correctness'] == 'correct'])}")
print(f"Accuracy: {len(df[df['correctness'] == 'correct']) / len(df) * 100:.2f}%")

# sort by correctness AND solution, save original IDS
df["id"] = df.index
df = df.sort_values(by=["correctness", "solution"])
df = df.reset_index(drop=True)
# with open("data/completions/starcoderbase-1b-completions_typeinf_analysis.csv", "w") as f:
#     # remove prompt column
#     df.drop(columns=["prompt"]).to_csv(f)

ds = datasets.Dataset.from_pandas(df)
# basename = os.path.basename(completions_file).replace(".json", "")
# ds.push_to_hub(f"franlucc/{basename}_typeinf_analysis")