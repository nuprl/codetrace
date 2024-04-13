import tree_sitter
from codetrace.utils import replace_between_bytes, get_captures, PY_PARSER, PY_LANGUAGE
import datasets
from collections import defaultdict
from vllm import LLM, SamplingParams
import random
from tqdm import tqdm
import pandas as pd
import re
import sys
from multiprocessing import cpu_count
from argparse import ArgumentParser
from typing import List, Tuple, Union, Callable
from dataclasses import dataclass
import random
import typing
import builtins
from collections import namedtuple
from pyminifier.minification import remove_comments_and_docstrings
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

PY_IDENTIFIER_QUERY = """((identifier) @name)""" # this also captures types
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
FUNCTION_NAME = """(function_definition name: ((identifier) @fname))"""
PY_VARIABLE_DECLARATION_QUERY = FUNCTION_PARAMS + FUNCTION_NAME

PY_ATTRIBUTE_IDENTIFIER_QUERY = """(attribute attribute: (identifier) @attr_id)"""

#NOTE: the following captures include colon and identifier (if present)
# eg. cap(n : int) = n : int, cap(-> n) = -> n
# needs postprocessing
PY_TYPE_ANNOTATIONS_QUERY = """((typed_parameter) @annotation)"""

RETURN_TYPES = """  [
        (function_definition return_type: 
            (type (identifier) @id))

]"""


class TreeSitterLocation:
    start_byte : int
    end_byte : int
    start_point : Tuple[int, int]
    end_point : Tuple[int, int]
    
    def __init__(self, tree_sitter_capture : Union[tree_sitter.Node, Tuple[tree_sitter.Node, str]]):
        if isinstance(tree_sitter_capture, tuple):
            tree_sitter_node, _ = tree_sitter_capture
        else:
            tree_sitter_node = tree_sitter_capture
        self.start_byte = tree_sitter_node.start_byte
        self.end_byte = tree_sitter_node.end_byte
        self.start_point = tree_sitter_node.start_point
        self.end_point = tree_sitter_node.end_point
        
        
    def __repr__(self):
        return f"""TreeSitterLocation(
            start_byte={self.start_byte},
            end_byte={self.end_byte},
            start_point={self.start_point},
            end_point={self.end_point}
    )"""

@dataclass
class Mutation:
    location : TreeSitterLocation
    byte_replacement : bytes
    prefix : Union[bytes, None] = None
    
    def __repr__(self):
        if self.prefix is not None:
            prefix = str(self.prefix)
        else:
            prefix = "None"
            
        return f"""Mutation(
        {self.location.__repr__()},
        replacement={str(self.byte_replacement)},
        prefix={prefix}
    )"""

def mutation_rename_vars(var_captures : List[Tuple[tree_sitter.Node,str]]) -> List[Mutation]:
    """
    Make mutations for renaming vraiables in VAR_CAPTURES.
    NOTE: new name cannot exist elsewhere in the program, must be different in format from type names.
    The format for vars this function uses is: __tmp{var_index}
    We assume the program does not naturally contain variables with this format
    """
    # map names to captures
    all_names = set([x[0].text for x in var_captures])
    # map name to new name
    name_to_new_name = {name : bytes(f"__tmp{i}","utf-8") for i, name in enumerate(all_names)}
    mutations = []
    for capture in var_captures:
        location = TreeSitterLocation(capture)
        replacement = name_to_new_name[capture[0].text]
        mutation = Mutation(location, replacement)
        mutations.append(mutation)
    return mutations


def needs_alias(typ: str):
    # if type is a builtin, needs alias
    return any([typ==str(t) for t in dir(builtins)+dir(typing)])
    
def mutation_rename_type(type_captures : List[Tuple[tree_sitter.Node,str]]) -> List[Mutation]:
    """
    Make mutations for renaming types. Assign a new name to each type in type_captures.
    If a type needs it, we create a new type alias for its renamed version.
    
    NOTE: new name cannot exist elsewhere in the program, must be different in format from variable names.
    We assume the program does not naturally contain types with format __typ{type_index}
    """
    # map names to captures
    all_names = set([x[0].text for x in type_captures])
    # map names to new names
    name_to_new_name = {name : bytes(f"__typ{i}","utf-8") for i, name in enumerate(all_names)}
    
    mutations = []
    do_not_prefix = set()
    for capture in type_captures:
        location = TreeSitterLocation(capture)
        replacement = name_to_new_name[capture[0].text]
        
        if needs_alias(capture[0]):
            # make new type alias
            prefix = replacement + b" = " + capture[0].text
        else:
            prefix = None
        mutation = Mutation(location, replacement, prefix)
        mutations.append(mutation)
    return mutations

def mutation_delete_annotation(annotation_captures : List[Tuple[tree_sitter.Node,str]])-> List[Mutation]:
    """
    Delete the type annotations from captures
    """
    mutations = []
    for capture in annotation_captures:
        location = TreeSitterLocation(capture)
        mutation = Mutation(location, b"")
        mutations.append(mutation)
    return mutations

def add_type_aliases_after_imports(code: bytes, type_aliases : List[bytes]) -> bytes:
    """
    Add type aliases to the prefix after the last import statement
    NOTE:we assume all imports are at the top of the file
    """
    type_aliases = b"\n".join(type_aliases) + "\n\n"
    captures = get_captures(code, IMPORT_STATEMENT_QUERY, [], "py")
    if len(captures) == 0:
        return type_aliases + code
    # find the last import statement
    last_import = max(captures, key=lambda x: x[0].end_byte)
    new_code = code[:last_import[0].end_byte] + b"\n" + type_aliases + code[last_import[0].end_byte:]
    return new_code

def apply_mutations(program : str, mutations : List[Mutation]) -> str:
    """
    Apply mutations to the program.
    NOTE: 
    - applies from bottom up in order to not disturb the byte offsets of other mutations
    - there's the issue that type rename mutations may be nested inside remove annotation mutations
        therefore, if a mutation is nested inside another mutation, keep only the parent mutation
    """
    # take care of nested mutations
    mutations = merge_nested_mutation(mutations)
    mutations.sort(key=lambda x: x.location.start_byte, reverse=True)
    
    byte_program = program.encode("utf-8")
    prefixes = []
    for mutation in mutations:
        byte_program = replace_between_bytes(byte_program, mutation.location.start_byte, mutation.location.end_byte, mutation.byte_replacement)
        if mutation.prefix is not None:
            prefixes.append(mutation.prefix)

    if len(prefixes) > 0:
        return add_type_aliases_after_imports(byte_program, prefixes).decode("utf-8")
    else:
        return byte_program.decode("utf-8")

def merge_nested_mutation(mutations : List[Mutation]) -> List[Mutation]:
    """
    Merge nested annotation mutations. 
    """
    mutations.sort(key=lambda x: x.location.start_byte, reverse=True)
    new_mutations = []
    # work in pairs, if next capture is a superset of the current one, skip curr
    for (curr, prev) in zip(mutations, mutations[1:]):
        if (curr.location.start_point[0] == prev.location.start_point[0] and 
            curr.location.start_point[1] >= prev.location.start_point[1] and
            curr.location.end_point[0] == prev.location.end_point[0] and
            curr.location.end_point[1] <= prev.location.end_point[1]):
            continue
        else:
            new_mutations.append(curr)
            
    # add the last capture in all cases
    new_mutations.append(mutations[-1])       
    return new_mutations

def random_mutate_sequential(
    program : str, 
    fim_type : str,
    mutations : List[Callable],
) -> str:
    """
    Apply random combination of mutations to the program.
    NOTE: does rename variables first, then rename types, then delete
    """
    new_program = program
    if mutation_rename_vars in mutations:
        p = random_mutate(new_program, fim_type, [mutation_rename_vars])
        if p != None:
            new_program = p
            
    if mutation_rename_type in mutations:
        p = random_mutate(new_program, fim_type, [mutation_rename_type])
        if p != None:
            new_program = p
            
    if mutation_delete_annotation in mutations:
        p = random_mutate(new_program, fim_type, [mutation_delete_annotation])
        if p != None:
            new_program = p
            
    return new_program


def postprocess_py_annotation(node_capture : Tuple[tree_sitter.Node, str],
                              target_char : bytes,
                              shift_amt : int) -> Tuple[tree_sitter.Node, str]:
    """
    Postprocess the annotation node by applying a shift to the node from the target character. 
    Captured annotations contain var id and type id, for example:
        n : int
    We want to extract only:
        : int
    Thus, need to shift node location + text
    """
    text = node_capture[0].text
    # find the index of the colon
    index = text.index(target_char)
    # count num bytes to shift
    shift = index + shift_amt
    # shift the node
    new_start_byte = node_capture[0].start_byte + shift
    new_start_point = (node_capture[0].start_point[0], node_capture[0].start_point[1] + shift)
    new_text = text[shift:]
    
    # edit node
    TSNodeAlias = namedtuple("TSNodeAlias", ["start_byte", "end_byte", "start_point", "end_point", "text"])
    new_node = TSNodeAlias(new_start_byte, node_capture[0].end_byte, new_start_point, node_capture[0].end_point, new_text)
    node_capture = (new_node, node_capture[1])
    assert node_capture[0].text == new_text, f"Text mismatch: {node_capture[0].text} != {new_text}"
    
    return node_capture

def postprocess_py_return_type(node_capture : Tuple[tree_sitter.Node, str], byte_program : bytes) -> Tuple[tree_sitter.Node, str]:
    """
    Return types in tree sitter don't include the ->, so we need to add it back
    """
    text = node_capture[0].text
    # find the first index of '->' starting from the end
    index = byte_program[:node_capture[0].start_byte].rfind(b"->")
    
    new_start_byte = index
    shift = index - node_capture[0].start_byte
    new_start_point = (node_capture[0].start_point[0], node_capture[0].start_point[1] + shift)
    new_text = byte_program[index:node_capture[0].end_byte]
    
    # edit node
    TSNodeAlias = namedtuple("TSNodeAlias", ["start_byte", "end_byte", "start_point", "end_point", "text"])
    new_node = TSNodeAlias(new_start_byte, node_capture[0].end_byte, new_start_point, node_capture[0].end_point, new_text)
    node_capture = (new_node, node_capture[1])
    assert node_capture[0].text == new_text, f"Text mismatch: {node_capture[0].text} != {new_text}"
    
    return node_capture


def random_mutate(
    program : str, 
    fim_type : str,
    mutations : List[Callable],
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
    program = program.replace("<FILL>", "_CodetraceSpecialPlaceholder_")
    
    # do not rename or delete these types
    types_blacklist = [bytes(fim_type,"utf-8"), 
                       bytes("_CodetraceSpecialPlaceholder_", "utf-8")]
    import_statements = get_captures(program, IMPORT_STATEMENT_QUERY, language="py")
    # -----------------------
    # get SELECT captures for target nodes that we can mutate
    program_bytes = bytes(program, "utf-8")
    tree = PY_PARSER.parse(program_bytes)

    var_rename_captures = get_captures(tree, PY_VARIABLE_DECLARATION_QUERY, language="py")
    return_types_captures = get_captures(tree, RETURN_TYPES, language="py")
    class_names = get_captures(tree, CLASS_NAMES, language="py")
    type_annotations_captures = get_captures(tree, PY_TYPE_ANNOTATIONS_QUERY, language="py")
    
    type_rename_captures = [postprocess_py_annotation(x, b":", 1) for x in type_annotations_captures] + class_names + return_types_captures
    remove_annotations_captures = [postprocess_py_annotation(x, b":", 0) for x in type_annotations_captures] 
    remove_annotations_captures +=  [postprocess_py_return_type(x, program_bytes) for x in return_types_captures]
    
    def select_random_subset(x):
        if debug_seed == -1 or len(x) == 0:
            return x
        n = random.randint(1, len(x))
        return random.sample(x, n)
    
    #  random subset of captures
    var_rename_captures = select_random_subset(var_rename_captures)
    type_rename_captures = select_random_subset(type_rename_captures)
    remove_annotations_captures = select_random_subset(remove_annotations_captures)
    
    # -----------------------
    # find ALL ADDITIONAL locations that contain targets
    var_rename_targets = set([x[0].text for x in var_rename_captures])
    type_rename_targets = set([x[0].text for x in type_rename_captures])
    
    all_id_captures = get_captures(tree, PY_IDENTIFIER_QUERY, language="py")
    all_attribute_ids = get_captures(tree, PY_ATTRIBUTE_IDENTIFIER_QUERY, language="py")
    attribute_names = set([x[0].text for x in all_attribute_ids])
    import_statement_names = b"\n".join([x[0].text for x in import_statements])
    var_rename_full_captures = [x for x in all_id_captures 
                                # rename all ids that match target
                                if x[0].text in var_rename_targets
                                # don't rename attributes
                                and not x[0].text in attribute_names #TODO: do we want to rename attributes?
                                # don't rename built-ins
                                and not x[0].text.decode("utf-8") in dir(builtins)+dir(typing)
                                # don't rename anything in import statements
                                and not x[0].text in import_statement_names
                                ]
    type_rename_full_captures = [x for x in all_id_captures
                                # rename all that match target
                                if x[0].text in type_rename_targets
                                # don't rename attributes
                                and not x[0].text in attribute_names #TODO: do we want to rename attributes?
                                # don't rename built-ins
                                and not x[0].text.decode("utf-8") in dir(builtins)+dir(typing)
                                # don't rename forbidden types
                                and x[0].text not in types_blacklist
                                # don't rename anything in import statements
                                and not x[0].text in import_statement_names
                                ]
    remove_annotations_captures = [x for x in remove_annotations_captures  
                                   if (x[0].text.replace(b":",b"").replace(b"->",b"").strip() not in types_blacklist
                                    # don't remove anything in import statements
                                    # TODO: relax this last filter?
                                    and x[0].text.replace(b":",b"").replace(b"->",b"").strip() not in import_statement_names)
                                   ]
    
    # -----------------------
    # Apply the selected mutations
    
    func_name_to_args = {
        "mutation_rename_vars" : var_rename_full_captures,
        "mutation_rename_type" : type_rename_full_captures,
        "mutation_delete_annotation" : remove_annotations_captures
    }
    # collects mutations
    all_mutations = []
    for m in mutations:
        all_mutations += m(func_name_to_args[m.__name__])
    
    if len(all_mutations) == 0:
        # bad run, return None
        if debug_seed is not None:
            return None, []
        return None
    
    # actually modify the program
    new_program = apply_mutations(program, all_mutations)
    if new_program == program:
        # no mods applied, return None
        return None
    
    # sometimes the placeholder can be deleted, for example in nested type annotations,
    # so here's a safety check
    if not "_CodetraceSpecialPlaceholder_" in new_program:
        return None
    
    new_program = new_program.replace("_CodetraceSpecialPlaceholder_", "<FILL>")
    
    if debug_seed is not None:
        return new_program, all_mutations
    
    return new_program


def iter_apply_random_mutations(iterable, mutations : List[Callable]):
    """
    Apply random combination of mutations
    """
    new_ds = []
    
    for i, ex in enumerate(iterable):
        new_program = None
        program = ex["fim_program"]
        fim_type = ex["fim_type"]
        
        tries = 0
        while new_program is None and tries < 10:
            tries += 1
            new_program = random_mutate_sequential(program, fim_type, mutations)
        if new_program is None:
            continue

        new_ds.append({"mutated_program": new_program, 
                       "mutations" : [m.__name__ for m in mutations], **ex})
    
    return new_ds

def remove_comments(program):
    try:
        res = remove_comments_and_docstrings(program).strip()
    except Exception as e:
        print(f"Error in removing comments: {e}")
        # sometimes parser fails, return empty string
        return ""
    return res
