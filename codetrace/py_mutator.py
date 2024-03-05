from tree_sitter import Language, Parser, Node
from typing import Generator, Set, List
from dataclasses import dataclass

Language.build_library(
    "build/my-languages.so",
    ["tree-sitter-python"],
)

PY_LANGUAGE = Language("build/my-languages.so", "python")

parser = Parser()
parser.set_language(PY_LANGUAGE)

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
    """
    captured_nodes = FUNCTION_PARAMS.captures(root_node)
    results = []
    for node, _ in captured_nodes:
        (start, end) = node.byte_range
        results.append(buffer[start:end].decode("utf-8"))
    return set(results)


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

    tree = parser.parse(buffer)

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


def mutations(code: str, skip_var: str) -> Generator[MutationResult, None, None]:
    """
    Produces all mutations of a file in depth-first order.

    code is the code to mutate.
    skip_var is the name of the variable to skip when mutating.
    """
    buffer = code.encode("utf-8")
    bound_vars_set = _get_bound_vars(buffer, parser.parse(buffer).root_node)
    bound_vars_set.remove(skip_var)
    bound_vars = list(bound_vars_set)
    yield from _mutations_rec(0, bound_vars, code, buffer)
