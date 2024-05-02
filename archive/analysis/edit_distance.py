from multiprocessing import cpu_count
import json
import datasets
import os
import pandas as pd
import sys
from pathlib import Path
from tqdm import tqdm 
from codetrace.fast_utils import get_batches_fast, batched_do_func

rouge = datasets.load_metric('rouge',trust_remote_code=True)
data_dir = sys.argv[1]
outfile = sys.argv[2]

def run_rouge(batch):
    results = []
    for f in batch:
        ds_pos = datasets.load_from_disk(f[0] + "/correct")
        ds_neg = datasets.load_from_disk(f[0] + "/incorrect")
        negative = ds_neg["fim_program"]
        positive = ds_pos["fim_program"]
        rouge_score = results = rouge.compute(predictions=negative,
                        references=positive)
        results.append({"name":f[0],"rouge_score":rouge_score, "pos_size":len(positive), "neg_size":len(negative)})
    return results

file_tups = []
for f in os.walk(data_dir, topdown=False):
    if "incorrect" in f[1]:
        file_tups.append(f)
        
batches = get_batches_fast(file_tups, cpu_count())
results = batched_do_func(batches, cpu_count(), run_rouge)

def yielder():
    for x in tqdm(results, desc="yielding", total=len(results)):
        yield x
        
ds = datasets.Dataset.from_generator(yielder)
ds.save_to_disk(outfile)