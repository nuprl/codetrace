"""
What we actually want to do:

1. For every bound variable X in a program, incrementally rename X to Y, where 
   Y is not  free in the scope of X.

It is almost impossible to do this for Python, so we instead simplify the
problem to this:

1. Calculate the set of all variable names bound in a program.
2. Incrementally rename each variable above to a variable that does not
   appear in the program text.

So, if X is bound twice, both occurrences will be renamed at once, which
is harmless.

However, there are a few complications:

1. A name may be bound by both a function and non-function binder. E.g.,
   import statements, class definitions, and function definitions all bind
   new names. Renaming an imported name is not semantics preserving. Renaming
   a function defined in a class (i.e., a method), requires renaming all
   method calls.

2. A Python program can dynamically modify the set of names in scope.

We assume that these do not occur in reasonable code.
"""
from tree_sitter import Node
from typing import Generator, Set, List
from dataclasses import dataclass
from .utils import PY_LANGUAGE, PY_PARSER
import datasets
import os
import glob
import pandas as pd
import random
from multiprocessing import cpu_count

# This query finds all the identifiers in a file.
IDENTIFIERS = PY_LANGUAGE.query("""(identifier) @id""")

# This query finds the parameters of function definitions. It does not do
# lambdas.
FUNCTION_PARAMS = PY_LANGUAGE.query(
    """
    [
        (function_definition parameters: 
            (parameters [ (identifier) @id (typed_parameter (identifier) @id) ]))

    ]
"""
)


@dataclass
class MutationResult:
    """
    Represents a mutation returned by the mutations generator below.
    """

    # The code before the last mutation.
    old_code: str
    # The fully mutated code.
    new_code: str
    # The number of mutations applied.
    num_mutations: int

    _cut: bool = False

    def cut(self):
        """
        The consumer of the mutations generator can call this method to indicate
        that there is no need to go deeper in the search for mutations.
        """
        self._cut = True


def _get_bound_vars(buffer: bytes, root_node: Node) -> Set[str]:
    """
    Returns the names of the variables bound by function definitions in a file.

    NOTE: We make the assumption that the names bound by functions are not
    also imported or defined at the top-level. If that is the case, renaming
    may not be semantics-preserving.
    """
    captured_nodes = FUNCTION_PARAMS.captures(root_node)
    results = []
    for node, _ in captured_nodes:
        (start, end) = node.byte_range
        results.append(buffer[start:end].decode("utf-8"))
    return set(results)


# There are ~114 types of non-terminals listed in the Python tree-sitter
# grammar:
#
# https://github.com/tree-sitter/tree-sitter-python/blob/master/src/node-types.json
#
# These are the statements that cannot contain references to variables bound
# by function definitions and lambdas. Of course, there are some truly wierd
# cases, for example:
#
#     def foo(c):
#         class c:
#             pass
#
# Moreover, code can dynamically modify the set of names in any scope. But,
# it should be safe to ignore these cases for most code.
NONVAR_STATEMENTS = [
    "import_statement",
    "import_from_statement",
    "import_prefix",
    "future_import_statement",
    "wildcard_import",
    "relative_import",
    "module", # What is this?
    "class_definition",
    "class_pattern",
]

# There are several other contexts where variables can appear. But, we are being
# safely lazy. It should be enough to check that we are in one the contexts
# below and not in the NONVAR_STATEMENTS contexts.
VAR_CONTEXTS = [
    "parameters",
    "function_definition"
]

def is_var_context(node: Node):
    """
    In a well-designed grammar, it would be obvious whether a name is a variable
    reference or not. Instead, we look at the encoding context to make a
    determination.
    """
    while node.parent:
        if node.type in VAR_CONTEXTS:
            return True
        if node.type in NONVAR_STATEMENTS:
            return False
        node = node.parent
    return True

def _rename_var(buffer: bytes, root_node: Node, old_name: str, new_name: str) -> str:
    """
    Renames all occurrences of the variable old_name to new_name.

    root_node must be the root of the whole file.
    buffer must be the byte buffer for the whole file.
    """
    result: List[str] = []
    last_offset_start = 0
    for node, _ in IDENTIFIERS.captures(root_node):
        (id_start, id_end) = node.byte_range
        this_name = buffer[id_start:id_end].decode("utf-8")
        if this_name != old_name:
            continue
        if not is_var_context(node):
            continue
        result.append(buffer[last_offset_start:id_start].decode("utf-8"))
        result.append(new_name)
        last_offset_start = id_end
    if last_offset_start < len(buffer):
        result.append(buffer[last_offset_start:].decode("utf-8"))
    return "".join(result)

def make_new_name(new_length : int) -> str:
    letters = "abcdefghijklmnopqrstuvwxyz"*10
    new_name = "".join(random.sample(letters, new_length))
    return new_name

def _mutations_rec(
    depth: int, bound_vars: List[str], old_code: str, buffer: bytes
) -> Generator[MutationResult, None, None]:
    """
    A generator that produces all mutations in a file in breath-first order.

    depth is the number of mutations applied so far.
    bound_vars is the list of variables that may be mutated.
    old_code is the current code prior to mutation.
    buffer is the byte buffer for old_code.
    """
    if len(bound_vars) == 0:
        return

    tree = PY_PARSER.parse(buffer)

    for var_ix, var in enumerate(bound_vars):
        new_name = make_new_name(len(var))
        new_code = _rename_var(buffer, tree.root_node, var, new_name)
        result = MutationResult(
            old_code=old_code, new_code=new_code, num_mutations=depth + 1
        )
        yield result

        # NOTE(arjun): This is a fascinating hack to cut off the search. The
        # consumer of this generator can call result.cut() to indicate that the
        # there is no need to go deeper in the search for mutations.
        if result._cut:
            continue

        yield from _mutations_rec(
            depth + 1, bound_vars[var_ix + 1 :], new_code, new_code.encode("utf-8")
        )

def rename_var(code: str, old_name: str, new_name: str) -> str:
    """
    Renames all occurrences of the variable old_name to new_name in code.
    """
    buffer = code.encode("utf-8")
    tree = PY_PARSER.parse(buffer)
    return _rename_var(buffer, tree.root_node, old_name, new_name)

def get_bound_vars(code: str) -> Set[str]:
    """
    Returns the names of the variables bound by function definitions in a file.
    """
    buffer = code.encode("utf-8")
    return _get_bound_vars(buffer, PY_PARSER.parse(buffer).root_node)

def mutations(code: str) -> Generator[MutationResult, None, None]:
    """
    Produces all mutations of a file in depth-first order.
    """
    buffer = code.encode("utf-8")
    bound_vars = list(_get_bound_vars(buffer, PY_PARSER.parse(buffer).root_node))
    yield from _mutations_rec(0, bound_vars, code, buffer)
    
def dataset_rename_vars(dataset: datasets.Dataset) -> datasets.Dataset:
    """
    For each example in the dataset, rename all variables incrementally
    """
    new_dataset = []
    
    os.makedirs("temp", exist_ok=True)
    
    def _mutate(x):
        new_dataset =[]
        for r in mutations(x["fim_program"]):
            if r.num_mutations > 10:
                r.cut()
            new_ex = x.copy()
            new_ex["renamed_fim_program"] = r.new_code
            new_ex["renamed_num"] = r.num_mutations
            new_dataset.append(new_ex)
        return {**x, "mutation_results": new_dataset}
    
    # shard ds and batch
    batch_size = 20
    start_idx = 240
    num_shards = len(dataset) // batch_size
    print(f"Sharding into {num_shards} shards")
    for i in range(start_idx, len(dataset), batch_size):
        shard = dataset.shard(num_shards=num_shards, index=i)
        shard = shard.map(_mutate, num_proc=batch_size)
        # save
        shard.save_to_disk(f"temp/temp_{i}")
    
    new_ds = []
    for f in glob.glob("temp/temp_*"):
        new_ds.append(datasets.load_from_disk(f))
    new_ds = datasets.concatenate_datasets(new_ds)
    print(new_ds)
    # unroll ds
    for ex in new_ds:
        for result in ex["mutation_results"]:
            new_dataset.append(result)
            
    new_dataset = datasets.Dataset.from_pandas(pd.DataFrame(new_dataset))
    return new_dataset