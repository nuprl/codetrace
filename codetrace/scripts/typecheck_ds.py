import os
import shutil
from pathlib import Path
import argparse
import datasets
import subprocess
from typing import List, Dict, Any, Tuple, Optional
import tempfile
import hashlib
from codetrace.fast_utils import batched_apply, make_batches
from codetrace.utils import load_dataset, save_dataset
from multiprocessing import cpu_count
import glob
import re
import shutil
import json
        
pyright_config = {
    "typeCheckingMode" : "basic",
    "reportMissingImports" : False,
    "analyzeUnannotatedFunctions" : False,
    "strictParameterNoneValue" : False,
    "reportMissingModuleSource" : False,
}

def log(path: str, data: Any):
    with open(path, "w") as f:
        f.write(data + "\n")

def found_error(lang:str, line:str) -> bool:
    if lang == "py":
        return ("- error:" in line and not '- error: Import "' in line and not "unknown import symbol" in line 
            and not "Position-only parameter not allowed after parameter that is not position-only" in line
            and not 'Expression of type "None" cannot be assigned to return type' in line
            and not 'is not a known member of "None"' in line 
            and not 'Imports from __future__ must be at the beginning of the file' in line)
    elif lang == "ts":
        return (": error" in line and not "Cannot find module" in line \
                and not "Invalid module name in augmentation" in line)
    else:
        raise NotImplementedError("Supported languages are py and ts")

def wrap(err:str, n_chars:int) -> str:
    res = ""
    for idx in range(0,len(err), n_chars):
        res += err[idx:idx+n_chars] + "\n"
    return res

def typecheck_py(dir:str) -> Tuple[str, List[str]]:
    files = list(glob.glob(f"{dir}/*.py"))
    commands = ["pyright", "-p", "pyright_config.json","*"]
    try:
        outs = subprocess.run(commands, cwd=dir, capture_output=True, timeout=300, text=True).stdout
    except Exception as e:
        print(f"Error in typechecking...{e}")
        outs = "\n".join([f"{f} - error: {e}" for f in files])
    return outs,files

def typecheck_ts(dir:str) -> Tuple[str, List[str]]:
    outs = []
    files = list(glob.glob(f"{dir}/*.ts"))
    commands = ["npx", "--cache", str(os.environ["NPM_PACKAGES"]), "ts-node", 
                "--compilerOptions", "{\"noImplicitAny\": false, \"strictNullChecks\": false}", 
                "--typeCheck"]
    for tsfile in files:
        tsfile = tsfile.split("/")[-1]
        try:
            out = subprocess.run(commands + [tsfile], cwd=dir, capture_output=True, timeout=120,text=True).stderr
            outs.append(out)
        except Exception as e:
            print(f"Error in typechecking...{e}")
            outs.append(f"{tsfile}: error {e}")

    return "\n".join(outs),files
        
# from https://github.com/nuprl/MultiPL-T/
def run_typechecker(dir:str, lang:str) -> Dict[str, Dict[str,str]]:
    if lang == "py":
        outs,files= typecheck_py(dir)
    elif lang == "ts":
        outs,files = typecheck_ts(dir)
    else:
        raise ValueError("Only ts and py langs supported")
        
    cur_file, cur_lines = "",""
    filemap = {}
    for _, line in enumerate(outs.split("\n")):
        filename_in_line = re.findall(r"^.*\.{lang}".format(lang=lang), line)
        if len(filename_in_line) == 1:
            filename_in_line = filename_in_line[0].split("/")[-1]
            # found new file
            if filename_in_line != cur_file and any([f.split('/')[-1] == filename_in_line for f in files]):
                # get new file
                filemap[filename_in_line] = {"count":0, "errors":[]}
                cur_file = filename_in_line
                cur_lines = ""
        
        if len(cur_file) > 0 and found_error(lang, line):
            cur_lines += line + "\n"
            filemap[cur_file]["count"] += 1
            filemap[cur_file]["errors"].append(line)
            
    for filename in files:
        if filename.split("/")[-1] not in filemap:
            filemap[filename.split("/")[-1]] = {"count":0, "errors":[]}
    return filemap

def _format_error_list(error_list: List[str])-> str:
    col_len = 80
    delim = f"\n#{'='*col_len}\n"
    res = ""
    for err in error_list:
        res += wrap(err,col_len).strip() + delim
    return res

def hash_string(input_string):
    sha256 = hashlib.sha256()
    sha256.update(input_string.encode('utf-8'))
    return sha256.hexdigest()
    
def typecheck_batch(
    examples: List[dict],
    colname: str, 
    lang:str,
    logdir:Optional[Path]=None
) -> List[dict]:
    new_ds = []
    hexsha_to_ex = {}
    
    if logdir:
        os.makedirs(logdir)

    with tempfile.TemporaryDirectory() as tempdir:
        for ex in examples:
            program = ex[colname].replace("<FILL>", ex["fim_type"])
            hexsha = hash_string(program)
            hexsha_to_ex[hexsha] = ex
            name = os.path.join(tempdir, hexsha + f".{lang}")
            with open(name, "w") as f:
                f.write(program)

        with open(f"{tempdir}/pyright_config.json", "w") as f:
            json.dump(pyright_config, f)

        # Run pyright in the temporary directory
        typecheck_map = run_typechecker(tempdir, lang)
        if typecheck_map is None:
            return []

        for i, (hexsha, dikt) in enumerate(typecheck_map.items()):
            num_errors = dikt["count"]
            error_list = dikt["errors"]
            new_ds.append({**hexsha_to_ex[hexsha.replace(f".{lang}","")], 
                           "typechecks": num_errors == 0,
                           "error_list": "\n".join(error_list)})
            if logdir:
                item = hexsha_to_ex[hexsha.replace(f".{lang}","")]
                original_prog,original_type = item["fim_program"], item["fim_type"]
                original = original_prog.replace("<FILL>", original_type)
                log(f"{logdir}/_original_{name.split('/')[-1]}", original)
                log(f"{logdir}/_mutated_{name.split('/')[-1]}", original)
                log(f"{logdir}/_errors_{name.split('/')[-1]}", _format_error_list(error_list))

    return new_ds

def main(
    ds: datasets.Dataset,
    outpath: str,
    **typechecker_args
):
    """
    For each ood_steering_ds/steering_ds in list-o-dirs, run parser. Collect % of parsing programs.
    """
    batches = make_batches(ds, cpu_count())
    result = batched_apply(batches, cpu_count(), typecheck_batch, **typechecker_args)
    ds_new = datasets.Dataset.from_list(result)
    print(ds_new)
    ds_new.save_to_disk(outpath)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-ds", type=str, required=True)
    parser.add_argument("--lang", choices=["py", "ts"], required=True)
    parser.add_argument("--output-ds", type=str, required=True)
    parser.add_argument("--column-name", type=str, required=True, help="column with fim program to typecheck")

    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--subset", type=str, default=None)
    parser.add_argument("--max-size", type=int, default=-1)
    parser.add_argument("--do-log", action="store_true")
    args = parser.parse_args()
            
    assert os.path.exists(Path(os.environ["NPM_PACKAGES"])), "Please pass a path to npm package location"
    
    logdir=None
    if args.do_log:
        # warn user it will overwrite previous logs
        print("[WARNING] Overwriting previous logs")
        logdir = f"/tmp/codetrace_logs/{args.lang}_log"
        if os.path.exists(logdir) and os.path.isdir(logdir):
            shutil.rmtree(logdir)
        os.makedirs(logdir, exist_ok=True)

    ds = load_dataset(args.input_ds, args.split, name=args.subset)
    print(ds)

    if args.max_size > -1:
        ds = ds.shuffle().select(range(args.max_size))

    main(ds, args.output_ds, colname=args.column_name, lang=args.lang, logdir=logdir)
