import os
from pathlib import Path
import argparse
import datasets
import pandas as pd
from codetrace.utils import PY_PARSER, TS_PARSER, placeholder_to_std_fmt, STARCODER_FIM
from tqdm import tqdm
import subprocess
from typing import List, Dict
import tempfile
import hashlib
from codetrace.fast_utils import batched_do_func
from multiprocessing import cpu_count
import glob
import random
import numpy as np

# from https://github.com/nuprl/MultiPL-T/
def run_typechecker(d, lang):
    if lang == "py":
        logdir = "pyright_log"
        try:
            outs = subprocess.run(
                ["pyright", "*"],
                cwd=d,
                capture_output=True,
                timeout=300,
                text=True,
            ).stdout
        except Exception as e:
            print("Error running pyright: ", e)
            return None
        
    elif lang == "ts":
        outs = []
        logdir = "tsc_log"
        for tsfile in glob.glob(f"{d}/*.ts"):
            tsfile = tsfile.split("/")[-1]
            try:
                out = subprocess.run(
                    ["npx", "--cache", str(os.environ["NPM_PACKAGES"]), "ts-node", "--typeCheck", tsfile],
                    cwd=d,
                    capture_output=True,
                    timeout=120,
                    text=True,
                ).stderr
                outs.append(out)
            except Exception as e:
                print("Error running tsnode: ", e)

        outs = "\n".join(outs)
    else:
        raise ValueError("Only ts and py langs supported")
    
    # save dir + log file
    os.makedirs(logdir, exist_ok=True)
    cur_file = ""
    cur_lines = ""
    filemap = {}
    lines = outs.split("\n")
    for i, line in enumerate(lines):
        if i == len(lines) - 2:
            break
        
        if line.startswith("  "):
            if ((lang == "py" and "- error:" in line and not '- error: Import "' in line)
                or (lang == "ts" and ": error" in line and not "Cannot find module" in line)):
                cur_lines += line + "\n"
                filemap[cur_file] += 1
        else:
            if len(cur_file) > 0 and filemap[cur_file] > 0:
                with open(f"{logdir}/{cur_file}".replace(f".{lang}", "") + ".txt", "w") as fp:
                    fp.write(cur_lines)
                with open(f"{d}/{cur_file}", "r") as fp:
                    contents = fp.read()
                with open(f"{logdir}/{cur_file}", "w") as fp:
                    fp.write(contents)
            file = line.split("/")[-1]
            filemap[file] = 0
            cur_file = file
            cur_lines = ""

    return filemap

    
def typecheck_batch(files: List[str], lang) -> Dict[str, str]:
    # Create a temporary directory using the tempfile module
    filemap: Dict[str, str] = {}
    prefix = f"/scratch/lucchetti.f/tmp/{lang}"
    hexsha = hashlib.sha256(bytes("".join(files), "utf-8")).hexdigest()
    tempdir = f"{prefix}/{hexsha}"
    os.makedirs(tempdir, exist_ok=True)
    for contents in files:
        hash_object = hashlib.sha1(bytes(contents, "utf8"))
        hex_dig = hash_object.hexdigest()
        filemap[hex_dig] = contents
        name = os.path.join(tempdir, hex_dig + f".{lang}")
        with open(name, "w") as f:
            f.write(contents)

    # Run pyright in the temporary directory
    typecheck_map = run_typechecker(tempdir, lang)
    if typecheck_map is None:
        return {}

    for contents, errors in typecheck_map.items():
        no_ext = contents.replace(f".{lang}", "")
        if errors == 0:
            continue

        if no_ext in filemap:
            del filemap[no_ext]

    return filemap

def get_node_types(node) -> str:
    s = node.type + "\n"
    for child in node.children:
        s += get_node_types(child)
    return s

def has_error(tree):
    node_types = get_node_types(tree.root_node)
    return "ERROR" in node_types

def multiproc_typecheck(programs, lang, batch_size=5):
    batches = [programs[i:i+batch_size] for i in range(0, len(programs), batch_size)]
    results = batched_do_func(batches, cpu_count(), typecheck_batch, lang=lang)
    return len(results)

def get_typecheck_ratio(programs, lang):
    """
    Dump programs in a temp dir.
    Run pyright on temp dir.
    Collect typecheck %
    """
    num_typecheck = multiproc_typecheck(programs, lang)
    return (num_typecheck / len(programs))

def get_parse_ratio(parser, programs, lang):
    prefix = f"/scratch/lucchetti.f/tmp/{lang}"
    parse_trees = [parser.parse(bytes(p, "utf-8")) for p in programs]
    error = list(filter(has_error, parse_trees))
    os.makedirs(f"{prefix}/log_parse_{lang}", exist_ok=True)
    for p in error:
        hexsha = hashlib.sha256(p.text).hexdigest()
        with open(f"{prefix}/log_parse_{lang}/{hexsha}.{lang}", "w") as f:
            f.write(p.text.decode("utf-8"))
    return  1 - (len(error) / len(parse_trees))

def process_dataset(subdir, dsname):
    res_ds = datasets.load_from_disk(f"{subdir}/{dsname}")
    programs = res_ds["fim_program"]
    if "<FILL>" in programs[0]:
        programs = [p.replace("<FILL>", res_ds[i]["fim_type"]) for i,p in enumerate(programs)]
    else:
        programs = [placeholder_to_std_fmt(p, STARCODER_FIM) for p in programs]
    return programs

def main(args):
    """
    For each ood_steering_ds/steering_ds in list-o-dirs, run parser. Collect % of parsing programs.
    """
    if args.lang == "py":
        parser = PY_PARSER
    else:
        parser = TS_PARSER
    
    data = []
    
    list_o_dirs = [d for d in args.list_o_dirs if (not "rand" in d and not "caa" in d and "fit" in d)]
    for subdir in tqdm(list_o_dirs, desc="processing subdirs"):
        print(subdir)
        if os.path.exists(Path(f"{subdir}/steering_results_ds")):
            programs = process_dataset(subdir, "steering_results_ds")
            if args.max_size > -1:
                programs = programs[:args.max_size]
            parse_ratio = get_parse_ratio(parser, programs, args.lang)
            typecheck_ratio = get_typecheck_ratio(programs, args.lang)
        else:
            parse_ratio = None
            typecheck_ratio = None
        
        if os.path.exists(Path(f"{subdir}/ood_steering_results_ds")):
            programs = process_dataset(subdir, "ood_steering_results_ds")
            if args.max_size > -1:
                programs = programs[:args.max_size]
            ood_parse_ratio = get_parse_ratio(parser, programs, args.lang)
            ood_typecheck_ratio = get_typecheck_ratio(programs, args.lang)
        else:
            ood_parse_ratio = None
            ood_typecheck_ratio = None
        
        data.append({"subdir": subdir, "parse_ratio": parse_ratio, "ood_parse_ratio": ood_parse_ratio,
                     "typecheck_ratio": typecheck_ratio, "ood_typecheck_ratio": ood_typecheck_ratio})
        
        df = pd.DataFrame(data)
        df.to_csv(args.outfile)
    
    df = pd.DataFrame(data)
    df.to_csv(args.outfile)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--list-o-dirs", type=str, nargs="+", required=True)
    parser.add_argument("--lang", choices=["py", "ts"], required=True)
    parser.add_argument("--outfile", type=str, required=True)
    parser.add_argument("--npm-location", type=str, default="/work/arjunguha-research-group/franlucc/.npm_packages")
    parser.add_argument("--max-size", type=int, default=-1)
    args = parser.parse_args()
    os.environ["NPM_PACKAGES"] = args.npm_location
    print(os.environ["NPM_PACKAGES"])
    main(args)
