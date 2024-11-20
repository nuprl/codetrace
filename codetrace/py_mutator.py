import tree_sitter
from codetrace.parsing_utils import (
    get_captures, is_in_capture_range, PY_PARSER
)
from codetrace.base_mutator import AbstractMutator, Mutation, TreeSitterLocation
import random
from typing import List, Tuple, Callable
import random
import typing
import builtins
from collections import namedtuple
"""
https://github.com/nvim-treesitter/nvim-treesitter/blob/master/queries/python/highlights.scm
Random mutation code.

Some considerations.

1. renaming to an arbitrary name (especially length)

the trick to being able to rename to any name is to accumulate
all the changes and apply them finally from end to start

2. each mutation method should produce different types of names to prevent overlap
and also for semantics
"""

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
IMPORT_STATEMENTS = [
    "import_statement",
    "import_from_statement",
    "import_prefix",
    "future_import_statement",
    "wildcard_import",
    "relative_import"
]
IMPORT_STATEMENT_QUERY = """
((import_statement) @import_statement)
((import_from_statement) @import_statement)
((import_prefix) @import_statement)
((future_import_statement) @import_statement)
((wildcard_import) @import_statement)
((relative_import) @import_statement)
"""

# # There are several other contexts where variables can appear. But, we are being
# # safely lazy. It should be enough to check that we are in one the contexts
# # below and not in the NONVAR_STATEMENTS contexts.
# VAR_CONTEXTS = [
#     "parameters",
#     "module",  # Top-level variable I believe
#     "function_definition",
# ]

# TYPED_IDENTIFIERS = PY_LANGUAGE.query("""(typed_parameter) @param""")


CLASS_NAMES = """(class_definition name: (identifier) @id)"""

####

PY_IDENTIFIER_QUERY = """((identifier) @id)""" # this also captures types
PY_TYPE_IDENTIFIER_QUERY = """[
  (typed_parameter type:
      (type (identifier) @id))
]"""
# This query finds the parameters of function definitions. It does not do
# lambdas.
FUNCTION_PARAMS = """
    [
        (function_definition parameters: 
            (parameters [ (identifier) @id (typed_parameter (identifier) @id) ]))

    ]
"""
FUNCTION_NAME = """(function_definition name: ((identifier) @id))"""
PY_VARIABLE_DECLARATION_QUERY = FUNCTION_PARAMS + FUNCTION_NAME

PY_ATTRIBUTE_IDENTIFIER_QUERY = """(attribute attribute: (identifier) @id)"""

#NOTE: the following captures include colon and identifier (if present)
# eg. cap(n : int) = n : int, cap(-> n) = -> n
# needs postprocessing
PY_TYPE_ANNOTATIONS_QUERY = """((typed_parameter) @annotation)"""

RETURN_TYPES = """ (function_definition return_type: (type (_) @id))"""

PY_TOPLEVEL_CHILDREN = "(module (_) @root_children)"

DummyTreeSitterNode = namedtuple("DummyTreeSitterNode", [
                                        "start_byte", "end_byte", 
                                        "start_point", "end_point", "text"])
class PyMutator(AbstractMutator):
    
    def is_import_statement(self, expr : tree_sitter.Node) -> bool:
        print(expr.type)
        return expr.type in IMPORT_STATEMENTS

    def needs_alias(self, typ: bytes, import_statements : bytes) -> bool:
        # if type is a builtin or typing, needs alias
        # if a type is in imports, needs alias
        return any([typ==bytes(t,"utf-8") for t in dir(builtins)+dir(typing)]) or typ in import_statements
    
    def format_capture(self, capture: tree_sitter.Node, prefix: bytes, **kwargs) -> str:
        if self.needs_alias(capture.text, kwargs["import_statement_names"]):
            # make new type alias
            prefix = prefix + b'" : TypeAlias = "' + capture.text + '"'
        else:
            prefix = None
        return prefix

    def add_type_aliases(self, code: bytes, type_aliases : List[bytes]) -> bytes:
        """
        Add type aliases to the prefix after the relevant import statement.
        We assume there are either no imports, thus just add aliases at top,
        or type aliases are imported/created, in which case we add them before the first
        time they are ever used (if imported, place alias AFTER import
        """
        import_typ_alias = b"from typing import TypeAlias\n"

        # add before the first expression
        expression = get_captures(code, PY_TOPLEVEL_CHILDREN, "py", "root_children")
        # find the relevant FIRST expression to insert alias before
        replacement_muts : List[Mutation] = []
        for type_alias in type_aliases:
            type_original = type_alias.rsplit(b'=')[-1].replace(b'"',b'').strip()
            
            for possible_match in expression:
                if type_original in possible_match.text:
                    if self.is_import_statement(possible_match):
                        replacement = possible_match.text + b"\n" + type_alias
                    else:
                        replacement = type_alias + b"\n" + possible_match.text

                    replacement_muts.append(Mutation(
                                    TreeSitterLocation(possible_match),
                                    byte_replacement=replacement
                                ))
                    break
        
        # none found, insert at top
        if len(replacement_muts) == 0:
            return import_typ_alias + b"\n".join(type_aliases) + b"\n" + code
        # note we can only do this because we did not specify prefix, otherwise recursive loop
        applied_muts = self.apply_mutations(code.decode("utf-8"), replacement_muts)
        return import_typ_alias + bytes(applied_muts, "utf-8")

    def add_program_prefix(self, byte_program: bytes, prefixes: List[bytes]) -> bytes:
        prefixes = list(set(prefixes))
        return self.add_type_aliases(byte_program, prefixes)

    def postprocess_type_annotation(
        self,
        node_capture: tree_sitter.Node,
        target_char: bytes, # ":"
        shift_amt : int
    ) -> DummyTreeSitterNode:
        """
        Postprocess the annotation node by applying a shift to the node from the target character. 
        Captured annotations contain var id and type id, for example:
            n : int
        We want to extract only:
            : int
        Thus, need to shift node location + text
        """
        text = node_capture.text
        # find the index of the colon
        index = text.index(target_char)
        # count num bytes to shift
        shift = index + shift_amt
        # shift the node
        new_start_byte = node_capture.start_byte + shift
        new_start_point = (node_capture.start_point[0], node_capture.start_point[1] + shift)
        new_text = text[shift:]
        
        # edit node
        new_node = DummyTreeSitterNode(new_start_byte, node_capture.end_byte, new_start_point, 
                                       node_capture.end_point, new_text)
        node_capture = new_node
        assert node_capture.text == new_text, f"Text mismatch: {node_capture.text} != {new_text}"
        return node_capture

    def postprocess_return_type(
        self,
        node_capture: tree_sitter.Node,
        byte_program : bytes
    ) -> DummyTreeSitterNode:
        """
        Return types in tree sitter don't include the ->, so we need to add it back
        """
        # find the first index of '->' starting from the end
        index = byte_program[:node_capture.start_byte].rfind(b"->")
        
        new_start_byte = index
        shift = index - node_capture.start_byte
        new_start_point = (node_capture.start_point[0], node_capture.start_point[1] + shift)
        new_text = byte_program[index:node_capture.end_byte]
        
        # edit node
        new_node = DummyTreeSitterNode(new_start_byte, node_capture.end_byte, new_start_point, 
                    node_capture.end_point, new_text)
        node_capture = new_node
        assert node_capture.text == new_text, f"Text mismatch: {node_capture.text} != {new_text}"
        return node_capture

    def random_mutate(
        self,
        program: str,
        fim_type: str,
        mutations: List[Callable],
        debug_seed : int = None
    ) -> str:
        """
        Apply random combination of mutations to the program.
        Can provide a random seed DEBUG_SEED for debugging.
        NOTE: if debug_seed is -1, this is a special case where we do not select a random subset but
        and run the full set instead (DEGUB only)
        """
        if debug_seed is not None:
            random.seed(debug_seed)
            
        # to prevent tree-sitter error:
        program = self.replace_placeholder(program)
        
        # -----------------------
        # get SELECT captures for target nodes that we can mutate
        program_bytes = bytes(program, "utf-8")
        tree = PY_PARSER.parse(program_bytes)

        var_rename_captures = get_captures(tree, PY_VARIABLE_DECLARATION_QUERY, "py", "id")
        return_types_captures = get_captures(tree, RETURN_TYPES, "py", "id")
        class_names = get_captures(tree, CLASS_NAMES, "py", "id")
        type_annotations_captures = get_captures(tree, PY_TYPE_ANNOTATIONS_QUERY, "py", "annotation")
        
        type_rename_captures = [self.postprocess_type_annotation(x, b":", 1) for x in type_annotations_captures] \
                        + class_names + return_types_captures
        remove_annotations_captures = [self.postprocess_type_annotation(x, b":", 0) for x in type_annotations_captures] 
        remove_annotations_captures +=  [self.postprocess_return_type(x, program_bytes) for x in return_types_captures]
        
        def select_random_subset(x):
            if debug_seed == -1 or len(x) == 0:
                return x
            n = random.randint(1, len(x))
            return random.sample(x, n)
        
        #  random subset of captures
        var_rename = select_random_subset(var_rename_captures)
        type_rename = select_random_subset(type_rename_captures)
        remove_annotations = select_random_subset(remove_annotations_captures)
        
        # -----------------------
        # find ALL ADDITIONAL locations that contain targets
        
        var_rename_all, type_rename_all, remove_annotations_all = self.find_all_other_locations_of_captures(
            program,
            fim_type,
            var_rename,
            type_rename,
            remove_annotations
        )
        
        # -----------------------
        # Apply random combinations of mutations
        new_program, all_mutations = self.mutate_captures(
            program,
            mutations,
            var_rename_all,
            type_rename_all,
            remove_annotations_all
        )
        
        if debug_seed is not None:
            return new_program, all_mutations
        
        return new_program

    def find_all_other_locations_of_captures(
        self,
        program:str,
        fim_type:str,
        var_rename_captures: List[tree_sitter.Node],
        type_rename_captures: List[tree_sitter.Node],
        remove_annotations_captures: List[tree_sitter.Node]
    ) -> Tuple[tree_sitter.Node]:
        var_rename_targets = set([x.text for x in var_rename_captures])
        type_rename_targets = set([x.text.strip() for x in type_rename_captures])
        
        # do not rename or delete these types
        types_blacklist = [bytes(fim_type,"utf-8"), self.placeholder()]
        import_statements = get_captures(program, IMPORT_STATEMENT_QUERY, "py", "import_statement")
        all_id_captures = get_captures(program, PY_IDENTIFIER_QUERY, "py", "id")
        all_attribute_ids = get_captures(program, PY_ATTRIBUTE_IDENTIFIER_QUERY, "py", "id")
        attribute_names = set([x.text for x in all_attribute_ids])
        import_statement_names = b"\n".join([x.text for x in import_statements])
        var_rename_full_captures = [
            x for x in all_id_captures 
            # rename all ids that match target
            if x.text in var_rename_targets
            # don't rename attributes
            and not x.text in attribute_names #TODO: do we want to rename attributes?
            # don't rename built-ins because no alias supported for vars
            and not x.text.decode("utf-8") in dir(builtins)+dir(typing)
            # don't rename anything in import statements because no alias supported for vars
            and not x.text in import_statement_names
        ]
        type_rename_full_captures = [
            x for x in all_id_captures
            # rename all that match target
            if x.text in type_rename_targets
            # don't rename attributes
            and not x.text in attribute_names #TODO: do we want to rename attributes?
            # don't rename forbidden types
            and x.text not in types_blacklist
            # don't rename if in range of import statements 
            # NOTE: we can rename text in import statements because of alias support, 
            # but not the actual imports
            and not is_in_capture_range(x, import_statements)
        ]
        remove_annotations_captures = [
            x for x in remove_annotations_captures  
                if (x.text.replace(b":",b"").replace(b"->",b"").strip() != 
                    self.placeholder())
        ]
        return var_rename_full_captures, type_rename_full_captures, remove_annotations_captures