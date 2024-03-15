import tree_sitter
from codetrace.utils import replace_between_bytes, get_captures, TS_PARSER
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
"""
Random mutation code.

Some considerations.

1. renaming to an arbitrary name (especially length)

the trick to being able to rename to any name is to accumulate
all the changes and apply them at the end from top to bottom.

2. each mutation method should produce different types of names to prevent overlap
and also for semantics
"""
TS_IDENTIFIER_QUERY = """((identifier) @name)"""
TS_TYPE_IDENTIFIER_QUERY = """((type_identifier) @name)"""
TS_PREDEFINED_TYPE_QUERY = """((predefined_type) @name)"""
 
TS_VARIABLE_DECLARATION_QUERY = """
(required_parameter pattern: (identifier) @func_param)
(variable_declarator (identifier) @var_declaration)
(function_declaration (identifier) @func_declaration)
"""

TS_QUERY_TYPE_ANNOTATIONS = """((type_annotation) @name)"""

TS_QUERY_PARAM_TYPES = """
(required_parameter pattern: (_) (type_annotation) @tp)
(optional_parameter pattern: (_) (type_annotation) @tp)
return_type: (type_annotation) @tp
"""

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
    Rename all identifiers in the program that are the same as the varname at the given location
    NOTE: new name cannot exist elsewhere in the program, must be different in format from type names
    format for vars: __tmp{var_index}
    We assume the program does not naturally contain variables with this format
    """
    # map names to new_names
    all_names = set([x[0].text for x in var_captures])
    name_to_new_name = {name : bytes(f"__tmp{ix}","utf-8") for ix, name in enumerate(all_names)}
    mutations = []
    for capture in var_captures:
        location = TreeSitterLocation(capture)
        replacement = name_to_new_name[capture[0].text]
        mutation = Mutation(location, replacement)
        mutations.append(mutation)
    return mutations

def needs_alias(node : tree_sitter.Node) -> bool:
    """
    Whether the node, when renamed, will need a type alias to be added to the program.
    Includes:
    - predefined types
    """
    return node.type == "predefined_type"
    
def mutation_rename_type(type_captures : List[Tuple[tree_sitter.Node,str]]) -> List[Mutation]:
    """
    Make mutations for renaming types. Assign a new name to each type in type_captures.
    If a type needs it, we create a new type alias for its renamed version.
    
    NOTE: new name cannot exist elsewhere in the program, must be different in format from variable names.
    We assume the program does not naturally contain types with format __typ{type_index}
    """
    # map names to new_names
    all_names = set([x[0].text for x in type_captures])
    name_to_new_name = {name : bytes(f"__typ{ix}","utf-8") for ix, name in enumerate(all_names)}
    
    mutations = []
    do_not_prefix = set()
    for capture in type_captures:
        location = TreeSitterLocation(capture)
        replacement = name_to_new_name[capture[0].text]
        
        if needs_alias(capture[0]) and capture[0].text not in do_not_prefix:
            # make new type alias
            prefix = b"type " + replacement + b" = " + capture[0].text + b";"
            # only make one alias for each predefined type
            do_not_prefix.add(capture[0].text)
        else:
            do_not_prefix.add(capture[0].text)
            prefix = None
            
        mutation = Mutation(location, replacement, prefix)
        mutations.append(mutation)
    return mutations

def mutation_delete_annotation(annotation_captures : List[Tuple[tree_sitter.Node,str]])-> List[Mutation]:
    """
    Delete the type annotation at the given location
    """
    mutations = []
    for capture in annotation_captures:
        location = TreeSitterLocation(capture)
        mutation = Mutation(location, b"")
        mutations.append(mutation)
    return mutations

def apply_mutations(program : str, mutations : List[Mutation]) -> str:
    """
    Apply mutations to the program. Sort mutations by end_byte.
    NOTE: 
    - applies top to bottom to not disturb the byte offsets of other mutations
    - there's the issue that type rename mutations may be nested inside remove annotation mutations
        therefore, if a type rename mutation is nested inside a remove annotation mutation, remove it first
    """
    mutations = merge_nested_mutation(mutations)
    byte_program = program.encode("utf-8")
    prefixes = []
    for mutation in mutations:
        byte_program = replace_between_bytes(byte_program, mutation.location.start_byte, mutation.location.end_byte, mutation.byte_replacement)
        if mutation.prefix is not None:
            prefixes.append(mutation.prefix)

    if len(prefixes) > 0:
        prefixes = "\n".join([p.decode("utf-8") for p in set(prefixes)]) + "\n\n"
        return prefixes + byte_program.decode("utf-8")
    else:
        return byte_program.decode("utf-8")

def merge_nested_mutation(mutations : List[Mutation]) -> List[Mutation]:
    """
    Merge nested annotation mutations. 
    """
    mutations.sort(key=lambda x: x.location.start_byte, reverse=True)
    new_mutations = []
    # work in pairs, if next capture is a superset of the current one, skip it
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

def random_mutate(
    program : str, 
    fim_type : str,
    mutations : List[Callable],
    debug_seed : int = None
) -> str:
    """
    Apply random combination of mutations to the program.
    NOTE: if debug_seed is -1, this is a special case where we do not select a random subset (DEGUB only)
    """
    if debug_seed is not None:
        random.seed(debug_seed)
        
    # to prevent tree-sitter error:
    program = program.replace("<FILL>", "_CodetraceSpecialPlaceholder_")
    
    # do not rename or delete these types
    types_blacklist = [bytes(fim_type,"utf-8"), 
                       bytes("_CodetraceSpecialPlaceholder_", "utf-8")]
    
    # -----------------------
    # get SELECT captures for target nodes that we can mutate
    tree = TS_PARSER.parse(bytes(program, "utf-8"))
    var_rename_captures = get_captures(tree, TS_VARIABLE_DECLARATION_QUERY, language="ts")
    type_rename_captures = get_captures(tree, TS_QUERY_TYPE_ANNOTATIONS, language="ts")
    remove_annotations_captures = get_captures(tree, TS_QUERY_PARAM_TYPES, language="ts")
    
    def select_random_subset(x):
        if debug_seed == -1:
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
    type_rename_targets = set([x[0].text.replace(b":",b"").strip() for x in type_rename_captures])
    
    all_id_captures = get_captures(tree, TS_IDENTIFIER_QUERY, language="ts")
    all_type_id_captures = get_captures(tree, 
                                        TS_TYPE_IDENTIFIER_QUERY + TS_PREDEFINED_TYPE_QUERY, 
                                        language="ts")
    
    var_rename_full_captures = [x for x in all_id_captures if x[0].text in var_rename_targets]
    type_rename_full_captures = [x for x in all_id_captures+all_type_id_captures 
                                    if (x[0].text in type_rename_targets 
                                        and x[0].text not in types_blacklist)
                                ]
    remove_annotations_captures = [x for x in remove_annotations_captures 
                                        if (x[0].text.replace(b":",b"").strip() not in types_blacklist)
                                ]
    
    # -----------------------
    # Apply the selected mutations
    
    func_name_to_args = {
        "mutation_rename_vars" : var_rename_full_captures,
        "mutation_rename_type" : type_rename_full_captures,
        "mutation_delete_annotation" : remove_annotations_captures
    }
    # actually runs the mutations
    all_mutations = []
    for m in mutations:
        all_mutations += m(func_name_to_args[m.__name__])
    
    if len(all_mutations) == 0:
        # bad run, try again
        if debug_seed is not None:
            return None, []
        return None
    
    # modify the program
    new_program = apply_mutations(program, all_mutations)
    assert "_CodetraceSpecialPlaceholder_" in new_program, "Placeholder was mistakenly renamed!"
    new_program = new_program.replace("_CodetraceSpecialPlaceholder_", "<FILL>")
    
    if debug_seed is not None:
        return new_program, all_mutations
    
    return new_program


def dataset_apply_random_mutations(dataset : datasets.Dataset, mutations : List[Callable]) -> datasets.Dataset:
    """
    Apply random combination of mutations
    """
    new_ds = []
    
    # 1. capture all possible mutation locations
    for i, ex in tqdm(enumerate(ds), desc="Mutating", total_len=len(ds)):
        new_program = None
        program = ex["fim_program"]
        fim_type = ex["fim_type"]
        while new_program is None:
            new_program = random_mutate(program, fim_type, mutations)

        new_ds.append({"mutated_program": new_program, 
                       "mutations" : [m.__name__ for m in mutations], **ex})
    
    new_ds = datasets.Dataset.from_pandas(pd.DataFrame(new_ds))
    return new_ds