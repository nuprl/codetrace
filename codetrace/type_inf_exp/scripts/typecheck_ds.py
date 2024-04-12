import os
import shutil
from pathlib import Path
import argparse
import datasets
import pandas as pd
from codetrace.utils import PY_PARSER, TS_PARSER, STARCODER_FIM
from tqdm import tqdm
import subprocess
from typing import List, Dict
import tempfile
import hashlib
from codetrace.fast_utils import batched_do_func, get_batches_fast
from multiprocessing import cpu_count
import glob
import random
import numpy as np
import re
import shutil

def log(path, data):
    with open(path, "w") as f:
        f.write(data + "\n")

# from https://github.com/nuprl/MultiPL-T/
def run_typechecker(d, lang, do_log=False):
    logdir = f"{lang}_log"
    if lang == "py":
        files = list(glob.glob(f"{d}/*.py"))
        try:
            outs = subprocess.run(
                ["pyright", "*"],
                cwd=d,
                capture_output=True,
                timeout=300,
                text=True,
            ).stdout
        except Exception as e:
            print("Error in typechecking...")
            outs = "\n".join([f"{f} - error: {e}" for f in files])
        
    elif lang == "ts":
        outs = []
        files = list(glob.glob(f"{d}/*.ts"))
        for tsfile in files:
            tsfile = tsfile.split("/")[-1]
            try:
                out = subprocess.run(
                    ["npx", "--cache", str(os.environ["NPM_PACKAGES"]), "ts-node", 
                     "--compilerOptions", "{\"noImplicitAny\": false, \"strictNullChecks\": false}", 
                     "--typeCheck", tsfile],
                    cwd=d,
                    capture_output=True,
                    timeout=120,
                    text=True,
                ).stderr
                outs.append(out)
            except Exception as e:
                print("Error in typechecking...")
                outs.append(f"{tsfile}: error {e}")

        outs = "\n".join(outs)
    else:
        raise ValueError("Only ts and py langs supported")
        
    cur_file = ""
    cur_lines = ""
    filemap = {}
    lines = outs.split("\n")
    regex = re.compile(f"^.*\.{lang}")
    for i, line in enumerate(lines):
        filename_in_line = re.findall(regex, line)
        if len(filename_in_line) == 1:
            filename_in_line = filename_in_line[0].split("/")[-1]
            # found new file
            if filename_in_line != cur_file and any([f.split('/')[-1] == filename_in_line for f in files]):
                # log any errors
                if do_log and len(cur_file) > 0 and filemap[cur_file] > 0:
                    with open(f"{d}/{cur_file}","r") as f:
                        content = f.read()
                    log(f"{logdir}/{cur_file}", content)
                    log(f"{logdir}/{cur_file.replace('.'+lang,'.log')}", cur_lines)
                # get new file
                filemap[filename_in_line] = 0
                cur_file = filename_in_line
                cur_lines = ""
        
        if len(cur_file) > 0  and lang == "py" and "- error:" in line and not '- error: Import "' in line:
            cur_lines += line + "\n"
            filemap[cur_file] += 1
        elif (len(cur_file) > 0 and lang == "ts" and ": error" in line and not "Cannot find module" in line
            and not "Invalid module name in augmentation" in line):
            cur_lines += line + "\n"
            filemap[cur_file] += 1
            
    for filename in files:
        if filename.split("/")[-1] not in filemap:
            filemap[filename.split("/")[-1]] = 0
    return filemap

    
def filter_typecheck_batch(examples: List[dict], colname, lang, do_log=False) -> List[dict]:
    filtered = []
    hexsha_to_ex = {}
    with tempfile.TemporaryDirectory() as tempdir:
        for ex in examples:
            program = ex[colname].replace("<FILL>", ex["fim_type"])
            hexsha = hashlib.sha1(bytes(program, "utf8")).hexdigest()
            hexsha_to_ex[hexsha] = ex
            name = os.path.join(tempdir, hexsha + f".{lang}")
            with open(name, "w") as f:
                f.write(program)

        # Run pyright in the temporary directory
        typecheck_map = run_typechecker(tempdir, lang, do_log=do_log)
        if typecheck_map is None:
            return []

        for hexsha, num_errors in typecheck_map.items():
            if num_errors == 0:
                filtered.append(hexsha.replace(f".{lang}",""))
    
    return [hexsha_to_ex[hexsha] for hexsha in filtered]


def main(args):
    """
    For each ood_steering_ds/steering_ds in list-o-dirs, run parser. Collect % of parsing programs.
    """
    if args.lang == "py":
        parser = PY_PARSER
    else:
        parser = TS_PARSER

    if args.local_dataset:
        ds = datasets.load_from_disk(args.dirname)
    else:
        ds = datasets.load_dataset(args.dirname, split="train")
        
    if args.max_size > -1:
        ds = ds.shuffle(42).select(range(args.max_size))

    batches = get_batches_fast(ds, len(ds), cpu_count())
    result = batched_do_func(batches, cpu_count(), filter_typecheck_batch, 
                             colname=args.column_name, lang=args.lang, do_log=args.do_log)
    def yielder():
        for ex in tqdm(result, desc="Yielding", total=len(result)):
            yield ex
    
    ds_new = datasets.Dataset.from_generator(yielder)
    print(ds_new)
    ds_new.push_to_hub(args.new_ds_name, private=True)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirname", type=str, required=True)
    parser.add_argument("--lang", choices=["py", "ts"], required=True)
    parser.add_argument("--new-ds-name", type=str, required=True)
    parser.add_argument("--column-name", type=str, required=True)
    
    parser.add_argument("--local-dataset", action="store_true", default=False)
    parser.add_argument("--npm-location", type=str, default="~/.npm_packages")
    parser.add_argument("--max-size", type=int, default=-1)
    parser.add_argument("--do-log", action="store_true", default=False)
    args = parser.parse_args()
            
    os.environ["NPM_PACKAGES"] = args.npm_location
    assert os.path.exists(Path(args.npm_location)), "Please pass a path to npm package location"
    
    if args.do_log:
        # warn user it will overwrite previous logs
        print("[WARNING] Overwriting previous logs")
        logdir = f"{args.lang}_log"
        if os.path.exists(logdir) and os.path.isdir(logdir):
            shutil.rmtree(logdir)
        os.makedirs(logdir, exist_ok=True)

    main(args)
