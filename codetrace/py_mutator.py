from tree_sitter import Node
from typing import Generator, Set, List, Tuple
from dataclasses import dataclass
from .utils import PY_LANGUAGE, PY_PARSER
from more_itertools import interleave_longest
import random


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
    # "class_definition",
    # "class_pattern",
]

# There are several other contexts where variables can appear. But, we are being
# safely lazy. It should be enough to check that we are in one the contexts
# below and not in the NONVAR_STATEMENTS contexts.
VAR_CONTEXTS = [
    "parameters",
    "module",  # Top-level variable I believe
    "function_definition",
]

TYPED_IDENTIFIERS = PY_LANGUAGE.query("""(typed_parameter) @param""")

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

CLASS_NAMES = PY_LANGUAGE.query("""(class_definition name: (identifier) @id)""")


@dataclass
class Edit:
    """
    A convenience class that represents an edit to a buffer. Without this,
    we would be producing integer triples with no indication of what each
    number means.
    """

    start_byte: int
    old_end_byte: int
    new_end_byte: int


@dataclass
class EditableNode:
    """
    TreeSitter's nodes are immutable, so this is a workaround. We also
    don't care about tracking nodes' line and column numbers. So this further
    simplifies the representation to only track their byte position.
    """

    start_byte: int
    end_byte: int
    # We have code that looks at the enclosing context. However, note that that
    # the positions in the context may get stale, since they are TreeSitter
    # nodes and not EditableNodes.
    parent: Node
    type: str

    def __repr__(self):
        return f"EditableNode({self.start_byte}, {self.end_byte}, ...)"

    @staticmethod
    def from_node(node: Node):
        return EditableNode(node.start_byte, node.end_byte, node.parent, node.type)

    def adjust(self, edit: Edit):
        if self.start_byte < edit.start_byte:
            # This node is before the edit, so no need to do anything.
            return
        byte_delta = edit.new_end_byte - edit.old_end_byte
        self.start_byte += byte_delta
        self.end_byte += byte_delta

    def contains_byte(self, byte: int) -> bool:
        """Test if the given byte is within the range of this node."""
        return self.start_byte <= byte < self.end_byte


def edit_nodes(edits: List[Edit], other_nodes: List[EditableNode]):
    for edit in edits:
        for node in other_nodes:
            node.adjust(edit)


def is_var_context(node: Node):
    """
    In a well-designed grammar, it would be obvious whether a name is a variable
    reference or not. Instead, we look at the enclosing context to make a
    determination.
    """
    while node.parent:
        if node.type in VAR_CONTEXTS:
            return True
        if node.type in NONVAR_STATEMENTS:
            return False
        node = node.parent
    return True


def remove_type_annotion(
    buffer: bytes, typed_parameter: EditableNode
) -> Tuple[bytes, List[Edit]]:
    assert typed_parameter.type == "typed_parameter"
    text = buffer[typed_parameter.start_byte : typed_parameter.end_byte].decode("utf-8")
    # The before_colon_text may include the comma
    before_colon_text = text.split(":", maxsplit=1)[0]
    var = before_colon_text.encode("utf-8")
    new_buffer = (
        buffer[: typed_parameter.start_byte] + var + buffer[typed_parameter.end_byte :]
    )
    edit = Edit(
        start_byte=typed_parameter.start_byte,
        old_end_byte=typed_parameter.end_byte,
        new_end_byte=typed_parameter.start_byte + len(var),
    )
    return new_buffer, [edit]


def rename_var(
    buffer: bytes, root: Node, old_var: bytes, new_var: bytes
) -> Tuple[bytes, List[Edit]]:
    assert root.type == "module"
    all_vars = [
        EditableNode.from_node(node) for (node, _) in IDENTIFIERS.captures(root)
    ]
    edits = []
    for ix in range(len(all_vars)):
        node = all_vars[ix]
        this_var = buffer[node.start_byte : node.end_byte]
        if this_var != old_var:
            continue
        if not is_var_context(node):
            continue
        buffer = buffer[: node.start_byte] + new_var + buffer[node.end_byte :]
        this_edit = Edit(
            start_byte=node.start_byte,
            old_end_byte=node.end_byte,
            new_end_byte=node.start_byte + len(new_var),
        )
        edits.append(this_edit)
        edit_nodes([this_edit], all_vars[ix + 1 :])
    return buffer, edits


def _get_bound_vars(buffer: bytes, root_node: Node) -> Set[bytes]:
    """
    Returns the names of the variables bound by function definitions in a file.

    NOTE: We make the assumption that the names bound by functions are not
    also imported or defined at the top-level. If that is the case, renaming
    may not be semantics-preserving.
    """
    captured_nodes = FUNCTION_PARAMS.captures(root_node)
    captured_nodes.extend(CLASS_NAMES.captures(root_node))
    results = []
    for node, _ in captured_nodes:
        (start, end) = node.byte_range
        results.append(buffer[start:end])
    return set(results)


def random_mutations(code: str, fixed_type_location: int, apply_all_mutations: bool) -> Generator[Tuple[int, str], None, None]:
    """
    Generate a sequence of random mutations to a Python program. Each successive
    mutation is to the previous mutation. The fixed_type_location is a byte offset
    to a type annotation that is *not* mutated.
    """
    buffer = code.encode("utf-8")
    tree = PY_PARSER.parse(buffer)
    root = tree.root_node

    all_vars = list(_get_bound_vars(buffer, root))

    adjusted_type_location = fixed_type_location

    all_type_annotations = [
        EditableNode.from_node(item[0]) for item in TYPED_IDENTIFIERS.captures(root)
    ]

    all_type_annotations = [
        node
        for node in all_type_annotations
        if not node.contains_byte(fixed_type_location)
    ]

    random.shuffle(all_vars)
    random.shuffle(all_type_annotations)

    next_var_index = 0
    max_mutations = len(all_vars) + len(all_type_annotations)
    for counter in range(max_mutations):
        index = random.randint(0, max_mutations - counter - 1)
        if index < len(all_vars):
            (buffer, edits) = rename_var(
                buffer,
                root,
                all_vars.pop(),
                f"__tmp{next_var_index}".encode("utf-8"),
            )
            next_var_index += 1
        else:
            (buffer, edits) = remove_type_annotion(buffer, all_type_annotations.pop())
        # Adjust locations
        edit_nodes(edits, all_type_annotations)
        for edit in edits:
            if edit.start_byte < adjusted_type_location:
                adjusted_type_location += edit.new_end_byte - edit.old_end_byte
        if not apply_all_mutations:
            yield (adjusted_type_location, buffer.decode("utf-8"))
        
    if apply_all_mutations:
        yield (adjusted_type_location, buffer.decode("utf-8"))
