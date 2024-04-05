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

# from https://github.com/nuprl/MultiPL-T/
# runs pyright in the given directory, returns stdout
# then, it logs the number of errors for each file
def run_pyright(d):
    try:
        outs = subprocess.run(
            ["pyright", "*"],
            cwd=d,
            capture_output=True,
            timeout=300,
            text=True,
        ).stdout
    except Exception as e:
        print(e)
        return None

    # save dir + log file
    os.makedirs("pyright_log", exist_ok=True)
    for f in glob.glob(f"{d}/*.py"):
        with open(f, "r") as fp:
            contents = fp.read()
        basename = f.split("/")[-1]
        with open(f"pyright_log/{basename}", "w") as fp:
            fp.write(contents)
    
    cur_file = ""
    cur_lines = ""
    filemap = {}
    lines = outs.split("\n")
    for i, line in enumerate(lines):
        if i == len(lines) - 2:
            break
        
        if line.startswith("  "):
            if "- error:" in line and not '- error: Import "' in line:
                cur_lines += line + "\n"
                filemap[cur_file] += 1
        else:
            with open(f"pyright_log/{cur_file}".replace(".py", "") + ".txt", "w") as fp:
                fp.write(cur_lines)
            file = line.split("/")[-1]
            filemap[file] = 0
            cur_file = file
            cur_lines = ""

    return filemap

def typecheck_batch(files: List[str]) -> Dict[str, str]:
    # Create a temporary directory using the tempfile module
    filemap: Dict[str, str] = {}
    with tempfile.TemporaryDirectory() as tempdir:
        for contents in files:
            hash_object = hashlib.sha1(bytes(contents, "utf8"))
            hex_dig = hash_object.hexdigest()
            filemap[hex_dig] = contents
            name = os.path.join(tempdir, hex_dig + ".py")
            with open(name, "w") as f:
                f.write(contents)

        # Run pyright in the temporary directory
        typecheck_map = run_pyright(tempdir)
        if typecheck_map is None:
            return {}

    for contents, errors in typecheck_map.items():
        no_py = contents.replace(".py", "")
        if errors == 0:
            continue

        if no_py in filemap:
            del filemap[no_py]

    # print(f"Pass rate: {len(filemap)}/{len(files)}")

    return filemap

def get_node_types(node) -> str:
    s = node.type + "\n"
    for child in node.children:
        s += get_node_types(child)
    return s

def has_error(tree):
    node_types = get_node_types(tree.root_node)
    return "ERROR" in node_types

def multiproc_typecheck(programs, batch_size=10):
    batches = [programs[i:i+batch_size] for i in range(0, len(programs), batch_size)]
    results = batched_do_func(batches, cpu_count(), typecheck_batch)
    return len(results)

def get_typecheck_ratio(programs, lang):
    """
    Dump programs in a temp dir.
    Run pyright on temp dir.
    Collect typecheck %
    TODO: make mutliproc
    """
    if lang == "py":
        num_typecheck = multiproc_typecheck(programs)
        return (num_typecheck / len(programs))
    else:
        raise NotImplementedError("Type check for ts not impl")

def process_dataset(subdir, dsname):
    res_ds = datasets.load_from_disk(f"{subdir}/{dsname}")
    programs = res_ds["fim_program"]
    if "<FILL>" in programs[0]:
        programs = [p.replace("<FILL>", res_ds[i]["fim_type"]) for i,p in enumerate(programs)]
    else:
        programs = [placeholder_to_std_fmt(p, STARCODER_FIM) for p in programs]
    return programs

def get_parse_ratio(parser, programs):
    parse_trees = [parser.parse(bytes(p, "utf-8")) for p in programs]
    num_error = list(filter(has_error, parse_trees))
    return  1 - (len(num_error) / len(parse_trees))

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
            parse_ratio = get_parse_ratio(parser, programs)
            typecheck_ratio = get_typecheck_ratio(programs, args.lang)
        else:
            parse_ratio = None
            typecheck_ratio = None
        
        if os.path.exists(Path(f"{subdir}/ood_steering_results_ds")):
            programs = process_dataset(subdir, "ood_steering_results_ds")
            ood_parse_ratio = get_parse_ratio(parser, programs)
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
    args = parser.parse_args()
    main(args)