import json
import datasets
import pandas as pd
from utils import *

# Load the dataset
dataset = datasets.load_dataset("franlucc/stenotype-eval-dataset-func-type-stripped-test", split="train")

prompts = []
solution = []
original = []
    
for ex in dataset:
    stripped_prog = ex["content_type_removed"]
    type_map = json.loads(ex["type_map"])
    for k,v in type_map.items():
        prompt = replace_bytes_at(stripped_prog, "<FILL>", int(k))
        prompts.append(prompt)
        solution.append(v)
        original.append(ex["content"])
        
df = pd.DataFrame({"prompt": prompts, "solution": solution, "original": original})
ds = datasets.Dataset.from_pandas(df)
ds.push_to_hub("franlucc/stenotype-typeinf-prompts")