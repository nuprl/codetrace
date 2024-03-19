import pandas
import glob
import sys
import os
from pathlib import Path
import json
import pandas as pd

# pass in path of directory with all experiment results in it
results_dir = sys.argv[1]
outfile = sys.argv[2]

def basename(x):
    return x.split("/")[-1]

"""
Create a csv summarizing results with following columns:
- experiment dir name
- args.evaldir
- args.tokens_to_patch
- args.layers_to_patch
- ood_eval_readme.json accuracy, num_success, total
- eval_readme.json accuracy, num_success, total
- sum of previous two success, total, resulting accuracy
"""
csv_rows = []

for subdir in glob.glob(f"{results_dir}/*"):
    if os.path.isdir(Path(subdir)):
        # get args, ood_eval_readme, eval_readme
        fargs = f"{subdir}/args_steering.json"
        feval_ood = f"{subdir}/ood_eval_readme.json"
        feval_res = f"{subdir}/eval_readme.json"
        
        args, eval_res, eval_ood = {},{},{}
        if os.path.exists(Path(fargs)):
            with open(fargs, "r") as f:
                args = json.load(f)
                
        if os.path.exists(Path(feval_ood)):
            with open(feval_ood, "r") as f:
                eval_ood = json.load(f)
                
        if os.path.exists(Path(feval_res)):
            with open(feval_res, "r") as f:
                eval_res = json.load(f)
        
        total_num_success = eval_res.get("num_success",0) + eval_ood.get("num_success", 0)
        total_count = eval_res.get("total",0) + eval_ood.get("total", 0)
        if total_count > 0:
            total_accuracy = total_num_success / total_count
        else:
            total_accuracy = 0 # file not found
            total_num_success = 0
            total_count = 0
        row = {
            "experiment_dir" : basename(subdir),
            "eval_dir": basename(args["evaldir"]),
            "ood_accuracy": eval_ood.get("accuracy", 0),
            "ood_num_success":eval_ood.get("num_success",0),
            "ood_total":eval_ood.get("total",0),
            "fit_accuracy": eval_res.get("accuracy",0),
            "fit_num_success": eval_res.get("num_success",0),
            "fit_total": eval_res.get("total",0),
            "total_num_success": total_num_success,
            "total_count": total_count,
            "total_accuracy": total_accuracy,
            "tokens_patched": args["tokens_to_patch"],
            "layers_patched": args["layers_to_patch"],
        }
        csv_rows.append(row)

df = pd.DataFrame(csv_rows)
df.to_csv(outfile)
    
    
        
    