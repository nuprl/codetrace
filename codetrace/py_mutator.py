from tree_sitter import Node
from typing import Generator, Set, List
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

def is_var_context(node: Node):
    """
    In a well-designed grammar, it would be obvious whether a name is a variable
    reference or not. Instead, we look enclosing statement to make that
    determination.
    """
    result = [] 
    while node.parent:
        if node.type in NONVAR_STATEMENTS:
            return False
        # print(PY_LANGUAGE.lib.(node.type))
        result.append(node.type)
        node = node.parent
    print(result)
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
        new_name = f"__tmp{depth}"
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

def mutations(code: str, skip_var: str) -> Generator[MutationResult, None, None]:
    """
    Produces all mutations of a file in depth-first order.

    code is the code to mutate.
    skip_var is the name of the variable to skip when mutating.
    """
    buffer = code.encode("utf-8")
    bound_vars_set = _get_bound_vars(buffer, PY_PARSER.parse(buffer).root_node)
    bound_vars_set.remove(skip_var)
    bound_vars = list(bound_vars_set)
    yield from _mutations_rec(0, bound_vars, code, buffer)
