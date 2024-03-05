"""
Module for renaming a set of variables in a tyepscript program
using tree-sitter

TODO: make sure not capturing enums properly
"""
import tree_sitter
from codetrace.utils import *
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

TS_IDENTIFIER_QUERY = """((identifier) @name)""" 
PY_IDENTIFIER_QUERY = f"""((identifier) @name (#not-match? @name "{get_builtins_regex('python')}"))"""

lang_to_id_query = {
    "typescript" : TS_IDENTIFIER_QUERY,
    "python" : PY_IDENTIFIER_QUERY,
    "ts" : TS_IDENTIFIER_QUERY,
    "py" : PY_IDENTIFIER_QUERY
}
    
def capture_varnames(tree, language : str = "typescript") -> dict[str, list[tree_sitter.Node]]:
    """
    Given a program, capture all the variable names and their locations
    as tree-sitter (start, end) points
    """
    idquery = lang_to_id_query[language]
    
    ignore_parents = []
    if language in ["python", "py"]:
        # TODO: fix this
        ignore_parents = [r"\bdotted_name\b",
                          r"type_.*",
                          r"\bfunction_definition\b",
                          r"\bdecorator\b"
                          r".*_type",
                          r"\btype\b", 
                          r"\battribute\b",
                          r"\bcall\b"]
        
    captures = get_captures(tree, idquery, ignore_parents, language)
    vars_to_locs = defaultdict(list)
    for c in captures:
        name = c[0].text
        vars_to_locs[name].append(c[0])
    return {k.decode("utf-8"): v for k,v in vars_to_locs.items()}


def rename_variable(
    program : bytes,
    new_name : str,
    var_locations : list[tree_sitter.Node]) -> bytes:
    """
    Rename a variable in a program at the given locations
    NOTE: only works when new_name is same length as varname
    """
    # replace each varname with a new name
    for capture in var_locations:
        program = replace_between_bytes(program, capture.start_byte, capture.end_byte, new_name)
        
    return program

def make_new_name(new_length : int, existing_names : set[str]) -> str | None:
    """
    Given a set of var captures and a variable name, make a new name for the variable that
    is not already in the program.
    """
    letters = "abcdefghijklmnopqrstuvwxyz"*10
    new_name = "".join(random.sample(letters, new_length))
    tries = 0
    while new_name in existing_names:
        tries += 1
        if tries > 10000:
            return None
        new_name = "".join(random.sample(letters, new_length))
    return new_name


def dataset_rename_vars(dataset: datasets.Dataset, language: str) -> datasets.Dataset:
    """
    For each example in the dataset, rename all variables incrementally
    """
    parser = lang_to_parser[language]
    lang = lang_to_builder[language]
    
    new_dataset = []
    
    for i,ex in enumerate(tqdm(dataset)):
        fim_program = ex["fim_program"]
        solution = ex["fim_type"]
        
        tree = parser.parse(bytes( fim_program, "utf8"))
        var_locs = capture_varnames(tree, language=language)
        
        names, newnames = set(var_locs.keys()), set()
        
        fim_program = tree.text
        
        for varname, locs in var_locs.items():
            new_name = make_new_name(len(locs[0].text), names)
            if new_name is None or varname[0].isupper() or varname == solution:
                # TODO: fix this. there's some tree-sitter bug where it's capturing types as vars
                continue
            
            fim_program_new = rename_variable(fim_program, new_name, locs)
            try:
                fim_program_new.decode("utf-8")
            except:
                continue
            fim_program = fim_program_new
            
            names.add(new_name)
            newnames.add((varname, new_name))
            
            # save old ex
            new_dataset.append({**ex,
                "renamed_fim_program" : fim_program.decode("utf-8"),
                "renamed_variables" : list(newnames),
                "renamed_percent" : len(newnames) / len(var_locs),
            })
            
        
    new_dataset = datasets.Dataset.from_pandas(pd.DataFrame(new_dataset))
    # drop columns "correct" and "overfull"
    new_dataset = new_dataset.remove_columns(["correct", "overfull"])
    return new_dataset
