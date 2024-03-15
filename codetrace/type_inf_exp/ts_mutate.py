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

@dataclass
class TreeSitterLocation:
    start_byte : int
    end_byte : int
    start_point : Tuple[int, int]
    end_point : Tuple[int, int]
    
@dataclass
class Mutation:
    location : TreeSitterLocation
    byte_replacement : bytes
    prefix : Union[bytes, None] = None

def mutation_rename_vars(var_captures : List[Tuple[tree_sitter.Node,str]]) -> List[Mutation]:
    """
    Rename all identifiers in the program that are the same as the varname at the given location
    NOTE: new name cannot exist elsewhere in the program, must be different in format from type names
    """
    mutations = []
    for capture in var_captures:
        location = TreeSitterLocation(capture[0].start_byte, capture[0].end_byte, capture[0].start_point, capture[0].end_point)
        replacement = "" # TODO: make new name for variable
        mutation = Mutation(location, replacement)
        mutations.append(mutation)
    return mutations

def mutation_rename_type(type_captures : List[Tuple[tree_sitter.Node,str]]) -> List[Mutation]:
    """
    Rename all type identifiers in the program that are the same as the typename at the given location.
    If type is a primitive or larger than 1 token, add a new type declaration wrapper around the new
    name.
    NOTE: new name cannot exist elsewhere in the program, must be different in format from variable names
    """
    mutations = []
    for capture in type_captures:
        location = TreeSitterLocation(capture[0].start_byte, capture[0].end_byte, capture[0].start_point, capture[0].end_point)
        replacement = "" # TODO
        prefix = "" # TODO this is the new type declaration
        mutation = Mutation(location, replacement, prefix)
        mutations.append(mutation)
    return mutations

def mutation_delete_annotation(annotation_captures : List[Tuple[tree_sitter.Node,str]])-> List[Mutation]:
    """
    Delete the type annotation at the given location
    """
    mutations = []
    for location in annotation_locations:
        replacement = ""
        mutation = Mutation(location, replacement)
        mutations.append(mutation)
    return mutations

def apply_mutations(program : str, mutations : List[Mutation]) -> str:
    """
    Apply mutations to the program. Sort mutations by end_byte and apply from top to bottom
    """
    mutations.sort(key=lambda x: x.location.end_byte, reverse=True)
    prefixes = []
    for mutation in mutations:
        program = replace_between_bytes(program, mutation.location.start_byte, mutation.location.end_byte, mutation.byte_replacement)
        if mutation.prefix is not None:
            prefixes.append(mutation.prefix)
    prefix = "\n".join(prefixes) + "\n\n"
    return prefix + program

def dataset_apply_random_mutations(dataset : datasets.Dataset, mutations : List[Callable]) -> datasets.Dataset:
    """
    Apply random combination of mutations
    """
    # TODO replace <FILL> placeholder with a parsing placeholder for tree sitter
    
    # TODO select a random subset of mutations, then a random subset of corresponding captures
    # NOTE: DO NOT select the fim_type for renaming - any other type is fair game
    # preprocess the subset, i.e. don't touch the fim_type
    
    # collect all mutations and sort them by end_byte
    # apply them from top to bottom
    
    # -----------------------
    new_ds = []
    raise NotImplementedError("Test me first!")
    
    # 1. capture all possible mutation locations
    for i, ex in tqdm(enumerate(ds), desc="Mutating", total_len=len(ds)):
        
        program = ex["fim_program"].replace("<FILL>", "_CodetraceSpecialPlaceholder_")
        
        types_blacklist = [bytes(ex["fim_type"],"utf-8"), bytes("_CodetraceSpecialPlaceholder_", "utf-8")]
        
        var_rename_captures = get_captures(program, TS_VARIABLE_DECLARATION_QUERY, language="ts")
        type_rename_captures = get_captures(program, TS_QUERY_TYPE_ANNOTATIONS, language="ts")
        remove_annotations_captures = get_captures(program, TS_QUERY_PARAM_TYPES, language="ts")
        
        def select_random_subset(x):
            n = random.randint(0, len(x))
            return random.sample(x, n)
        
        var_rename_captures = select_random_subset(var_rename_captures)
        type_rename_captures = select_random_subset(type_rename_captures)
        remove_annotations_captures = select_random_subset(remove_annotations_captures)
        
        # find all additional locations that contain targets
        var_rename_targets = set([x[0].text for x in var_rename_captures])
        type_rename_targets = set([x[0].text for x in type_rename_captures])
        
        var_rename_full_captures = get_captures(program, TS_IDENTIFIER_QUERY, language="ts")
        type_rename_full_captures = get_captures(program, TS_TYPE_IDENTIFIER_QUERY, language="ts")
        
        var_rename_full_captures = [x for x in var_rename_full_captures if x[0].text in var_rename_targets]
        type_rename_full_captures = [x for x in type_rename_full_captures if (x[0].text in type_rename_targets and
                                                                              x[0].text not in types_blacklist)]
        
        # -----------------------
        # 2. apply select mutations
        func_name_to_args = {
            "mutation_rename_vars" : var_rename_full_captures,
            "mutation_rename_type" : type_rename_full_captures,
            "mutation_delete_annotation" : remove_annotations_captures
        }
        all_mutations = []
        for m in mutations:
            all_mutations += m(func_name_to_args[m.__name__])
        
        new_program = apply_mutations(program, all_mutations)
        new_ds.append({"mutated_program": new_program, "mutations" : [m.__name__ for m in mutations], **ex})
    
    new_ds = datasets.Dataset.from_pandas(pd.DataFrame(new_ds))
    return new_ds