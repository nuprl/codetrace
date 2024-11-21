import datasets
import itertools as it
from codetrace.scripts.typecheck_ds import typecheck_batch
from codetrace.fast_utils import batched_apply, make_batches
from codetrace.py_mutator import (
    PyMutator,
    PY_TYPE_ANNOTATIONS_QUERY,
    RETURN_TYPES as PY_RETURN_TYPES,
    PY_IDENTIFIER_QUERY,
    IMPORT_STATEMENT_QUERY,
    DummyTreeSitterNode
)
from tqdm import tqdm
from codetrace.base_mutator import Mutation, TreeSitterLocation,MutationFn
from codetrace.ts_mutator import TsMutator
import os
from typing import Union,Tuple,List,Dict
from codetrace.parsing_utils import get_captures

def _all_subsets(muts: List[MutationFn]) -> List[List[MutationFn]]:
    subsets = []
    for r in range(1, len(muts) + 1):
        subsets.extend(it.combinations(muts, r))
    subsets.append(tuple(muts))
    return [list(subset) for subset in subsets]

def test_all_subsets():
    output = _all_subsets([1,2,3])
    expected = [[1],[2],[3],[1,2],[2,3],[3,1], [1,2,3]]
    expected = [frozenset(x) for x in expected]
    output = [frozenset(x) for x in output]
    assert set(output) == set(expected)

def _mutate_dataset(
    items: List[Tuple[str, str]],
    mutator: Union[PyMutator, TsMutator],
    mutations: List[MutationFn]
) -> List[Dict[str,str]]:
    mutated_program_and_type = []
    for (prog,ftype) in items:
        mutated = mutator.random_mutate_ordered_by_type(prog, ftype, mutations)
        if mutated:
            mutated_program_and_type.append(
                {"mutated_program": mutated, "fim_type": ftype}
            )
    return mutated_program_and_type

def is_within_range(number, target, error):
    return target - number <= error

def colored_message(message, color):
    # ANSI escape codes for colored text (Red color for warning)
    reset = '\033[0m'
    print(f"{color}{message}{reset}\n")

def test_mutate_ds(lang:str,mutator:Union[TsMutator, PyMutator],ds: datasets.IterableDataset):
    # check that for N random mutations on dataset,
    # at least THRESHOLD typecheck
    THRESHOLD = 0.7
    NUM_EXAMPLES = 1000
    NUM_PROC = 30
    ERROR_RANGE = 0.05
    for mutations in tqdm(_all_subsets(
            [mutator.rename_types, mutator.rename_vars, mutator.delete_annotations]),
            "Testing all combinations of muts"):
        mutation_names = [m.__name__ for m in mutations]
        print("Testing:", mutation_names)
        items = next(ds.iter(NUM_EXAMPLES*2))
        programs, fim_types = items["fim_program"], items["fim_type"]
        items = list(zip(programs, fim_types))
        batches = make_batches(items, NUM_PROC)
        mutated_prog_and_type = batched_apply(batches, NUM_PROC, 
                                    _mutate_dataset, mutator=mutator, mutations = mutations,
                                    desc="Mutating")
        assert len(mutated_prog_and_type) > NUM_EXAMPLES
        mutated_prog_and_type = mutated_prog_and_type[:NUM_EXAMPLES]

        # typecheck
        items_that_typecheck = typecheck_batch(mutated_prog_and_type, "mutated_program", "py")

        mean = len(items_that_typecheck)/NUM_EXAMPLES
        message = f"[{lang}] {mutation_names}, {len(items_that_typecheck)}/{NUM_EXAMPLES}={mean:.2f}"

        if not is_within_range(mean, THRESHOLD, ERROR_RANGE):
            colored_message("[FAILED] "+message, '\033[91m') # red
        else:
            colored_message("[SUCC] "+message, '\033[92m') #green

def test_py_mutate_ds():
    # check that for N random mutations on dataset,
    # at least THRESHOLD typecheck
    ds = datasets.load_dataset("nuprl-staging/py_typeinf_fim", split="train", 
                               streaming=True).shuffle()
    test_mutate_ds("PY",PyMutator(),ds)

def test_ts_mutate_ds():
    # check that for N random mutations on dataset,
    # at least THRESHOLD typecheck
    ds = datasets.load_dataset("nuprl-staging/ts_typeinf_fim", split="train", 
                               streaming=True).shuffle()
    test_mutate_ds("TS",TsMutator(),ds)

if __name__ == "__main__":
    test_all_subsets()
    test_py_mutate_ds()
    test_ts_mutate_ds()