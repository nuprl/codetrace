import datasets
from codetrace.type_inf_exp.build_dataset import py_remove_annotations
from multiprocessing import cpu_count
import json
import argparse
import pandas as pd
from codetrace.utils import std_to_placeholder_fmt, STARCODER_FIM
import hashlib

# ds = datasets.load_dataset("franlucc/fim_items_10k", split="train")
# fim_placeholder="_$FILL"
# # make a fim_program and fim_type column
# ds = ds.map(lambda x: {"fim_program": x["prefix"] + fim_placeholder + x["suffix"], 
#                        "fim_type": x["middle"].strip(), **x}, num_proc=cpu_count())

# # map py_remove_annotations to fim_program
# ds = ds.map(lambda x: {"fim_program": py_remove_annotations(x["fim_program"], fim_placeholder)}, num_proc=cpu_count())

# # replace with standard placeholder
# ds = ds.map(lambda x: {"fim_program": x["fim_program"].replace(fim_placeholder, "<FILL>")}, num_proc=cpu_count())
# # save
# ds.push_to_hub("franlucc/fim_items_10k_prompt_vanilla")

parser = argparse.ArgumentParser()
parser.add_argument("--input_jsonl", type=str, required=True)
parser.add_argument("--output_ds", type=str, required=True)
args = parser.parse_args()

with open(args.input_jsonl,  "r") as f:
    data = [json.loads(l) for l in f.readlines()]
    print(data[0].keys())
    
ds = datasets.Dataset.from_pandas(pd.DataFrame(data))
ds = ds.rename_columns({"positive": "fim_program", "target": "fim_type", "negative":"renamed_fim_program"})
ds = ds.map(lambda x: {**x, "fim_program": std_to_placeholder_fmt(x["fim_program"], STARCODER_FIM),
                        "renamed_fim_program": std_to_placeholder_fmt(x["renamed_fim_program"], STARCODER_FIM),
                        "hexsha": hashlib.sha1(x["fim_program"].encode("utf-8")).hexdigest()})
ds.push_to_hub(args.output_ds)