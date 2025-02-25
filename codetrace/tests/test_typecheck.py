import datasets
import itertools as it
from codetrace.scripts.typecheck_ds import (
    run_typechecker, typecheck_py, typecheck_ts, multiproc_typecheck,found_error, found_error_in_line
)
from codetrace.fast_utils import batched_apply, make_batches
from codetrace.utils import print_color
from codetrace.py_mutator import PyMutator
from pathlib import Path
from tqdm import tqdm
from codetrace.base_mutator import MutationFn
from codetrace.ts_mutator import TsMutator
import os
from typing import Union,Tuple,List,Dict,Any
import uuid
CWD = os.path.abspath(Path(__file__).parent)

def write(content:str, file:str):
    with open(file, "w") as fp:
        fp.write(content)

def read(file:str) -> str:
    with open(file, "r") as fp:
        return fp.read()

def test_typecheck_py_single():
    prog1 = read(f"{CWD}/test_programs/test_no_typecheck.py")
    adir = f"/tmp/{uuid.uuid1()}"
    os.makedirs(adir)
    write(prog1, f"{adir}/prog1.py")
    
    filemap = run_typechecker(adir, "py", verbose=True, timeout=10, disable_tqdm=False)
    typecheck_output = typecheck_py(adir)
    assert "error" in typecheck_output["prog1"], typecheck_output["prog1"]
    assert filemap["prog1"] != None, filemap["prog1"]

def test_found_error():
    stderr = '''/tmp/afb8fb50-b6e4-11ef-8fee-79366ddce352/prog2.py
      /tmp/afb8fb50-b6e4-11ef-8fee-79366ddce352/prog2.py:2:14 - error: "__typ0" is not defined (reportUndefinedVariable)
      1 error, 0 warnings, 0 informations 
      '''
    assert found_error_in_line("py",stderr.split("\n")[1])
    assert found_error("py",stderr)

def test_typecheck_py():
    # correct, ignore import
    prog1 = """
from .soup import ultra_soup

def test(wohoo: int
)-> bool:
    print('weeeee')
    return True
"""
    # incorrect, undefined type error
    prog2 = """
def test(n : __typ0):
    return n
"""
    # correct
    prog3 = """
from typing import TypeAlias
__typ0 : TypeAlias = "str"
def test(n : __typ0):
    return n
"""
    adir = f"/tmp/{uuid.uuid1()}"
    os.makedirs(adir)
    write(prog1, f"{adir}/prog1.py")
    write(prog2, f"{adir}/prog2.py")
    write(prog3, f"{adir}/prog3.py")
    
    filemap = run_typechecker(adir, "py", verbose=True, timeout=10, disable_tqdm=False)
    typecheck_output = typecheck_py(adir)
    assert filemap["prog1"] == None, filemap["prog1"]
    assert "error" in typecheck_output["prog3"] and filemap["prog3"] == None, filemap["prog3"]
    assert filemap["prog2"] and "error" in filemap["prog2"], typecheck_output["prog3"]

def test_typecheck_ts():
    prog1 = read(f"{CWD}/test_programs/after_type_rename.ts").replace("<FILL>", "string")
    prog2 = """
public getCount(): number {
    return __typ0;
}
"""
    prog3 = """
import { someFunction } from 'nonexistent-module';
import * as fs from 'fs';
import * as path from 'path';

function countFilesInDir(directory: string): number {
    const files = fs.readdirSync(directory);
    return files.filter(file => fs.statSync(path.join(directory, file)).isFile()).length;
}
"""
    adir = f"/tmp/{uuid.uuid1()}"
    os.makedirs(adir)
    write(prog1, f"{adir}/prog1.ts")
    write(prog2, f"{adir}/prog2.ts")
    write(prog3, f"{adir}/prog3.ts")
    
    filemap = run_typechecker(adir, "ts", verbose=True, timeout=10, disable_tqdm=False)
    typecheck_output = typecheck_ts(adir)
    assert filemap["prog1"] == None, filemap["prog1"]
    assert "error" in typecheck_output["prog3"] and filemap["prog3"] == None, filemap["prog3"]
    assert "error" in filemap["prog2"], filemap["prog2"]


def _all_subsets(muts: List[MutationFn]) -> List[List[MutationFn]]:
    subsets = []
    for r in range(1, len(muts) + 1):
        subsets.extend(it.combinations(muts, r))
    return [list(subset) for subset in subsets]

def test_all_subsets():
    output = _all_subsets([1,2,3])
    expected = [[1],[2],[3],[1,2],[2,3],[1,3], [1,2,3]]
    output.sort()
    expected.sort()
    assert output == expected, f"{output} != {expected}"

def _get_mutator(lang: str) -> Union[PyMutator, TsMutator]:
    if lang == "py":
        return PyMutator()
    else:
        return TsMutator()

def _mutate_dataset(
    items: List[Tuple[str, str]],
    lang: str,
    mutations: List[str]
) -> List[Dict[str,str]]:
    mutator = _get_mutator(lang)
    mutated_program_and_type = []
    for (prog,ftype) in items:
        mutated = mutator.random_mutate_ordered_by_type(prog, ftype, mutations)
        if mutated:
            mutated_program_and_type.append(
                {"mutated_program": mutated, "fim_type": ftype, "fim_program": prog}
            )
    return mutated_program_and_type

def is_within_range(number, target, error):
    return target - number <= error

def dedup(data: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    """
    Necessary because same program can have <FILL> in different places,
    but when this is substituted with label, it may be identical to another
    mutated program with another fim location
    """
    seen = set()
    dedup_data = []
    for item in data:
        prog = item["mutated_program"].replace("<FILL>", item["fim_type"])
        if not prog in seen:
            dedup_data.append(item)
            seen.add(prog)
    return dedup_data

def test_mutate_ds(lang:str, ds: datasets.IterableDataset, logdir=None):
    # check that for N random mutations on dataset,
    # at least THRESHOLD typecheck
    THRESHOLD = 0.7
    NUM_EXAMPLES = 1000
    NUM_PROC = 30
    ERROR_RANGE = 0.05
    
    for mutations in tqdm(_all_subsets(
            ["rename_types", "rename_vars", "delete_annotations"]),
            "Testing all combinations of muts"):

        print("Testing:", mutations)

        items = next(ds.iter(NUM_EXAMPLES*3))
        programs, fim_types = items["fim_program"], items["fim_type"]
        items = list(zip(programs, fim_types))

        batches = make_batches(items, NUM_PROC)
        mutated_prog_and_type = batched_apply(batches, NUM_PROC, 
                                    _mutate_dataset, lang=lang, mutations = mutations,
                                    desc="Mutating")
        
        # get all unique full programs
        mutated_prog_and_type = dedup(mutated_prog_and_type)
        print(f"Len mutated {len(mutated_prog_and_type)}\n")
        assert len(mutated_prog_and_type) > NUM_EXAMPLES, len(mutated_prog_and_type)
        mutated_prog_and_type = mutated_prog_and_type[:NUM_EXAMPLES]
        
        # typecheck
        new_ds = multiproc_typecheck(mutated_prog_and_type, NUM_PROC,
                                     colname="mutated_program", lang=lang, logdir=logdir)
        new_df = datasets.Dataset.from_list(new_ds).to_pandas()
        items_that_typecheck = new_df[new_df["typechecks"]]
        assert len(new_df) == NUM_EXAMPLES, len(new_df)

        mean = len(items_that_typecheck)/NUM_EXAMPLES
        message = f"[{lang.upper()}]{mutations},{len(items_that_typecheck)}/{NUM_EXAMPLES}={mean:.2f}"

        if not is_within_range(mean, THRESHOLD, ERROR_RANGE):
            print_color("[FAILED] "+message, "red")
        else:
            print_color("[SUCC] "+message, "green")

def test_py_mutate_ds():
    # check that for N random mutations on dataset,
    # at least THRESHOLD typecheck
    ds = datasets.load_dataset("nuprl-staging/py_typeinf_fim", split="train", 
                               streaming=True).shuffle(buffer_size=2000)
    test_mutate_ds("py",ds, logdir=None)

def test_ts_mutate_ds():
    # check that for N random mutations on dataset,
    # at least THRESHOLD typecheck
    ds = datasets.load_dataset("nuprl-staging/ts_typeinf_fim", split="train", 
                               streaming=True).shuffle(buffer_size=2000)
    test_mutate_ds("ts",ds, logdir=None)

if __name__ == "__main__":
    assert os.path.exists(Path(os.environ["NPM_PACKAGES"])), \
        "Please set 'NPM_PACKAGES' env var to npm package location"
    test_found_error()
    test_all_subsets()
    test_typecheck_py()
    test_typecheck_ts()
    test_py_mutate_ds()
    test_ts_mutate_ds()
    test_typecheck_py_single()