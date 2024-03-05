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
from typing import Generator, Set, List, Tuple
from dataclasses import dataclass
from .utils import PY_LANGUAGE, PY_PARSER

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
    # New target index.
    new_target_index: int

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

def _rename_var(buffer: bytes, target_index: int, root_node: Node, old_name: str, new_name: str) -> Tuple[int, str]:
    """
    Renames all occurrences of the variable old_name to new_name.

    root_node must be the root of the whole file.
    buffer must be the byte buffer for the whole file.
    """
    result: List[str] = []
    last_offset_start = 0
    target_index_offset = 0
    name_size_delta = len(new_name) - len(old_name)
    for node, _ in IDENTIFIERS.captures(root_node):
        (id_start, id_end) = node.byte_range
        this_name = buffer[id_start:id_end].decode("utf-8")
        if this_name != old_name:
            continue
        if not is_var_context(node):
            continue
        result.append(buffer[last_offset_start:id_start].decode("utf-8"))
        result.append(new_name)
        if id_start <= target_index:
            target_index_offset += name_size_delta
        last_offset_start = id_end
    if last_offset_start < len(buffer):
        result.append(buffer[last_offset_start:].decode("utf-8"))
    return (target_index + target_index_offset, "".join(result))


def _mutations_rec(
    depth: int, target_index: int, bound_vars: List[str], old_code: str, buffer: bytes
) -> Generator[MutationResult, None, None]:
    """
    A generator that produces all mutations in a file in breath-first order.

    depth is the number of mutations applied so far.
    target_index is an offset into old_code that we are tracking. As variables
      are renamed, this offset is updated to account for a change in the length
      of the program.
    bound_vars is the list of variables that may be mutated.
    old_code is the current code prior to mutation.
    buffer is the byte buffer for old_code.
    """
    if len(bound_vars) == 0:
        return

    tree = PY_PARSER.parse(buffer)

    for var_ix, var in enumerate(bound_vars):
        new_name = f"__tmp{depth}"
        new_target_index, new_code = _rename_var(buffer, target_index, tree.root_node, var, new_name)
        result = MutationResult(
            old_code=old_code, new_code=new_code, num_mutations=depth + 1,
            new_target_index=new_target_index
        )
        yield result

        # NOTE(arjun): This is a fascinating hack to cut off the search. The
        # consumer of this generator can call result.cut() to indicate that the
        # there is no need to go deeper in the search for mutations.
        if result._cut:
            continue

        yield from _mutations_rec(
            depth + 1, new_target_index, bound_vars[var_ix + 1 :], new_code, new_code.encode("utf-8")
        )

def rename_var(code: str, old_name: str, new_name: str) -> str:
    """
    Renames all occurrences of the variable old_name to new_name in code.
    """
    buffer = code.encode("utf-8")
    tree = PY_PARSER.parse(buffer)
    return _rename_var(buffer, 0, tree.root_node, old_name, new_name)[1]


def rename_var_with_index(code: str, target_index: int, old_name: str, new_name: str) -> Tuple[int, str]:
    """
    Renames all occurrences of the variable old_name to new_name in code.
    """
    buffer = code.encode("utf-8")
    tree = PY_PARSER.parse(buffer)
    return _rename_var(buffer, target_index, tree.root_node, old_name, new_name)


def get_bound_vars(code: str) -> Set[str]:
    """
    Returns the names of the variables bound by function definitions in a file.
    """
    buffer = code.encode("utf-8")
    return _get_bound_vars(buffer, PY_PARSER.parse(buffer).root_node)

def mutations(code: str, target_index: int) -> Generator[MutationResult, None, None]:
    """
    Produces all mutations of a file in depth-first order.
    """
    buffer = code.encode("utf-8")
    bound_vars = list(_get_bound_vars(buffer, PY_PARSER.parse(buffer).root_node))
    yield from _mutations_rec(0, target_index, bound_vars, code, buffer)
