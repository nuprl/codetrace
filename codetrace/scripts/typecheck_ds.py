import asyncio
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
from codetrace.utils import load_dataset
from multiprocessing import cpu_count
import glob
import shutil
from tqdm import tqdm
from codetrace.scripts.bounded_subprocess_async import run as bounded_run
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
        return ("- error:" in line \
            and not '- error: Import "' in line \
            and not "unknown import symbol" in line 
            and not "Position-only parameter not allowed after parameter that is not position-only" in line
            and not 'Expression of type "None" cannot be assigned to return type' in line
            and not 'is not a known member of "None"' in line 
            and not 'Imports from __future__ must be at the beginning of the file' in line)
    elif lang == "ts":
        return (": error" in line \
                and not "Cannot find module" in line \
                and not "Invalid module name in augmentation" in line)
    else:
        raise NotImplementedError("Supported languages are py and ts")


def typecheck_py(dir:str, verbose=True, timeout=120, disable_tqdm=True) -> Dict[str,str]:
    files = list(glob.glob(f"{dir}/*.py"))
    file_to_error = {}
    with open(f"{dir}/pyright_config.json", "w") as f:
        json.dump(pyright_config, f)

    commands = ["pyright", "-p", "pyright_config.json"]
    for pyfile in tqdm(files, disable=disable_tqdm):
        pyfile = pyfile.split("/")[-1]
        try:
            out = asyncio.run(bounded_run(commands + [pyfile], cwd=dir, max_output_size=4096,
                                timeout_seconds=timeout))
            stderr = out.stdout + "\n" + out.stderr
            if out.timeout:
                raise ValueError("Subprocess timed out!")
            if out.exit_code != 0 and not "error" in stderr:
                raise ValueError(f"Missing stderr but exitcode was non-zero: {stderr}")
            
            file_to_error[pyfile.replace(".py","")] = stderr
        except Exception as e:
            if verbose:
                print(f"Error in typechecking...{e}")
            file_to_error[pyfile.replace(".py","")] = "{pyfile} - error: {e}"

    return file_to_error

def typecheck_ts(dir:str, verbose=True, timeout=300, disable_tqdm=True) -> Dict[str,str]:
    file_to_error = {}
    files = list(glob.glob(f"{dir}/*.ts"))
    commands = ["npx", "--cache", str(os.environ["NPM_PACKAGES"]), "ts-node", 
                "--compilerOptions", "{\"noImplicitAny\": false, \"strictNullChecks\": false}", 
                "--typeCheck"]
    for tsfile in tqdm(files, disable=disable_tqdm):
        tsfile = tsfile.split("/")[-1]
        try:
            out = asyncio.run(bounded_run(commands + [tsfile], cwd=dir, max_output_size=4096, 
                              timeout_seconds=timeout))
            stderr = out.stdout + "\n" + out.stderr
            if out.timeout:
                raise ValueError("Subprocess timed out!")
            if out.exit_code != 0 and not "error" in stderr:
                raise ValueError(f"Missing stderr but exitcode was non-zero: {stderr}")
            
            file_to_error[tsfile.replace(".ts","")] = stderr
        except Exception as e:
            if verbose:
                print(f"Error in typechecking...{e}")
            file_to_error[tsfile.replace(".ts","")] = f"{tsfile}: error {e}"

    return file_to_error
        
# from https://github.com/nuprl/MultiPL-T/
def run_typechecker(dir:str, lang:str, **kwargs) -> Dict[str,str]:
    if lang == "py":
        file_to_error= typecheck_py(dir, **kwargs)
    elif lang == "ts":
        file_to_error = typecheck_ts(dir, **kwargs)
    else:
        raise ValueError("Only ts and py langs supported")
        
    filemap = {}
    for file,stderr in file_to_error.items():
        if found_error(lang, stderr):
            filemap[file] = stderr
        else:
            filemap[file] = None

    return filemap

def hash_string(input_string):
    sha256 = hashlib.sha256()
    sha256.update(input_string.encode('utf-8'))
    return sha256.hexdigest()
    
def typecheck_batch(
    examples: List[dict],
    colname: str, 
    lang:str,
    logdir:Optional[Path]=None,
    **typechecker_kwargs
) -> List[dict]:
    new_ds = []
    pid_to_ex = {}
    
    if logdir:
        os.makedirs(logdir)

    with tempfile.TemporaryDirectory() as tempdir:
        for ex in examples:
            # add id
            program = ex[colname].replace("<FILL>", ex["fim_type"])
            pid = hash_string(program)
            pid_to_ex[pid] = ex
            name = os.path.join(tempdir, f"{pid}.{lang}")
            with open(name, "w") as f:
                f.write(program)

        # Run pyright in the temporary directory
        typecheck_map = run_typechecker(tempdir, lang, **typechecker_kwargs)
        if typecheck_map  == {}:
            return []

    for _, (fname, errors) in enumerate(typecheck_map.items()):
        item = pid_to_ex[fname]
        new_ds.append({**item, 
                        "typechecks": errors is None,
                        "errors": errors})
        if logdir:
            original_prog,original_type = item["fim_program"], item["fim_type"]
            original = original_prog.replace("<FILL>", original_type)
            log(f"{logdir}/_original_{name.split('/')[-1]}", original)
            log(f"{logdir}/_mutated_{name.split('/')[-1]}", item[colname])
            log(f"{logdir}/_errors_{name.split('/')[-1]}", errors)

    return new_ds

def multiproc_typecheck(data: List[Dict[str,Any]],nproc, **typechecker_args):
    batches = make_batches(data, nproc)
    result = batched_apply(batches, nproc, typecheck_batch, desc="Typechecking", **typechecker_args)
    return result

def main(
    ds: datasets.Dataset,
    outpath: str,
    **typechecker_args
):
    """
    For each ood_steering_ds/steering_ds in list-o-dirs, run parser. Collect % of parsing programs.
    """
    result = multiproc_typecheck(ds, cpu_count(), **typechecker_args)
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
