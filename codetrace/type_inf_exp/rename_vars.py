"""
Module for renaming a set of variables in a tyepscript program
using tree-sitter
"""
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

TS_IDENTIFIER_QUERY = """((identifier) @name)""" 
TS_VARIABLE_DECLARATION_QUERY = """
(required_parameter pattern: (identifier) @func_param)
(variable_declarator (identifier) @var_declaration)
(function_declaration (identifier) @func_declaration)
""" # ignore optional parameters for now because of `?`

# Node types:
#     https://github.com/tree-sitter/tree-sitter-typescript/blob/master/typescript/src/grammar.json
    
    
def capture_identifiers(tree) -> dict[str, Tuple[tree_sitter.Node, str]]:
    """
    Given a program, capture all the variable names and the tree-sitter nodes
    """
    captures = get_captures(tree, TS_IDENTIFIER_QUERY,language="ts")
    vars_to_node = defaultdict(list)
    for c in captures:
        name = c[0].text.decode("utf-8")
        vars_to_node[name].append(c)
    return vars_to_node


def rename_variable(
    program : bytes,
    new_name : str,
    var_nodes : list[tree_sitter.Node]) -> bytes:
    """
    Rename a variable in a program at the given locations
    NOTE: only works when new_name is same length as varname
    """ 
    for capture in var_nodes:
        program = replace_between_bytes(program, capture[0].start_byte, capture[0].end_byte, new_name)
        
    return program

def make_new_name(new_length : int, existing_names : set[str]) -> Union[str, None]:
    """
    Given a set of var captures and a variable name, make a new name for the variable that
    is not already in the program.
    """
    letters = "abcdefghijklmnopqrstuvwxyz"*10 #these are all 1 byte characters
    new_name = "".join(random.sample(letters, new_length))
    tries = 0
    while new_name in existing_names:
        tries += 1
        if tries > 10000:
            return None
        new_name = "".join(random.sample(letters, new_length))
    return new_name


def dataset_incremental_rename_vars(dataset: datasets.Dataset) -> datasets.Dataset:
    """
    For each example in the dataset, rename all variables incrementally
    """
    new_dataset = []
    
    for i,ex in enumerate(tqdm(dataset)):
        fim_program = ex["fim_program"]
        solution = ex["fim_type"]
        # replace <FILL> placeholder with a parsing placeholder for tree sitter
        fim_program = fim_program.replace("<FILL>", "_CodetraceSpecialPlaceholder_")
        
        tree = TS_PARSER.parse(bytes( fim_program, "utf8"))
        
        # get target variable names to rename ONLY from these variable declarations
        # the identifiers whose parents are in this set will be kept
        target_var_captures = get_captures(tree, TS_VARIABLE_DECLARATION_QUERY, language="ts")
        target_names = {c[0].text.decode("utf-8") for c in target_var_captures}
        
        all_ids_dict = capture_identifiers(tree)
        existing_names = set(all_ids_dict.keys())
        newnames = set()
        
        # get all other locations with the same variable names as target
        target_id_nodes = {k:v for k,v in all_ids_dict.items() if k in set(target_names)}
        
        fim_program = tree.text
        
        for varname, nodes in target_id_nodes.items():
            # len in bytes
            name_len = len(varname.encode("utf-8"))
            new_name = make_new_name(name_len, existing_names)
            
            if new_name is None:
                continue
            
            fim_program_new = rename_variable(fim_program, new_name, nodes)

            fim_program = fim_program_new
            
            existing_names.add(new_name)
            newnames.add((varname, new_name))
            
            decoded_fim_program = fim_program.decode("utf-8", errors="ignore")
            assert "_CodetraceSpecialPlaceholder_" in decoded_fim_program, "Placeholder Fill was accidentally renamed: {}".format(decoded_fim_program)
            new_dataset.append({**ex,
                "renamed_fim_program" : decoded_fim_program.replace("_CodetraceSpecialPlaceholder_", "<FILL>"),
                "renamed_variables" : list(newnames),
                "renamed_percent" : len(newnames) / len(target_id_nodes),
            })
            
        
    new_dataset = datasets.Dataset.from_pandas(pd.DataFrame(new_dataset))
    # drop columns "correct" and "overfull", no longer valid
    new_dataset = new_dataset.remove_columns(["correct", "overfull"])
    return new_dataset

"""
Random mutation code starts here.

Some considerations.

1. renaming to an arbitrary name (especially length)

the trick to being able to rename to any name is to accumulate
all the changes and apply them at the end from top to bottom.

2. each mutation method should produce different types of names to prevent overlap
and also for semantics
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

def mutation_rename_vars(var_locations : List[TreeSitterLocation]) -> List[Mutation]:
    """
    Rename all identifiers in the program that are the same as the varname at the given location
    """
    mutations = []
    for location in var_locations:
        replacement = "" # TODO: make new name for variable
        mutation = Mutation(location, replacement)
        mutations.append(mutation)
    return mutations

def mutation_rename_type(type_location : TreeSitterLocation, typename : str) -> List[Mutation]:
    """
    Rename all type identifiers in the program that are the same as the typename at the given location.
    If type is a primitive or larger than 1 token, add a new type declaration wrapper around the new
    name.
    """
    mutations = []
    for location in type_location:
        replacement = "" # TODO
        prefix = "" # TODO this is the new type declaration
        mutation = Mutation(location, replacement, prefix)
        mutations.append(mutation)
    return mutations

def mutation_delete_annotation(program, annotation_locations)-> List[Mutation]:
    """
    Delete the type annotation at the given location
    """
    mutations = []
    for location in annotation_locations:
        replacement = ""
        mutation = Mutation(location, replacement)
        mutations.append(mutation)
    return mutations

def dataset_apply_random_mutations(dataset : datasets.Dataset, mutations : List[Callable]) -> datasets.Dataset:
    """
    Apply random combination of mutations
    """
    # TODO replace <FILL> placeholder with a parsing placeholder for tree sitter
    
    # TODO select a random subset of mutations, then a random subset of corresponding captures
    # preprocess the subset, i.e. don't touch the fim_type
    
    # collect all mutations and sort them by end_byte
    # apply them from top to bottom
    pass
