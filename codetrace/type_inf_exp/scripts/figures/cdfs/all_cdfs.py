from codetrace.type_inf_exp.scripts.plot_results import plot_cdf
import glob
import sys
import os
import pandas as pd
import json

dirs = sys.argv[1]
print(dirs)
for f in glob.glob(f"{dirs}/**/**/ood_eval_readme.json"):
    outfile_name = f.split("/")[-2] + "_ood.pdf"
    if "caa" in outfile_name:
        continue
    print(f"Saving {outfile_name}")
    data = []
    with open(f, "r") as f:
        d = json.load(f)
        data += d["results_per_type"]
    data = pd.DataFrame(data)
    plot_cdf(data, outfile_name)
    
for f in glob.glob(f"{dirs}/**/**/eval_readme.json"):
    outfile_name = f.split("/")[-2] + "_steering.pdf"
    if "caa" in outfile_name:
        continue
    print(f"Saving {outfile_name}")
    data = []
    with open(f, "r") as f:
        d = json.load(f)
        data += d["results_per_type"]
    data = pd.DataFrame(data)
    plot_cdf(data, outfile_name)