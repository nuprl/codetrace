"""
take stenotype-eval-dataset and remove all type annotations but keep track of them
"""
from codetrace.utils import *
import datasets
import pandas as pd
import json
from ast import literal_eval
from typing import Tuple
import re
import tempfile
from tqdm import tqdm

TS_QUERY_ALL_TYPES = """((type_annotation) @name)"""
TS_QUERY_FUNC_TYPES = """
(required_parameter pattern: (_) (type_annotation) @tp)
(optional_parameter pattern: (_) (type_annotation) @tp)
return_type: (type_annotation) @tp
"""

"""
The following functions remove all type annotations but one <FILL> fim type
"""

def make_typeinf_prompts(
    dataset : datasets.Dataset, 
    query_str : str
    ) -> datasets.Dataset:
    """
    Make a dataset with all type annotations removed and a prompt for each type annotation
    """
    new_ds = []
    for ex in tqdm(dataset):
        prompts = fim_remove_types(ex["content"], query_str)
        for p in prompts:
            ex = ex.copy()
            ex["fim_program"] = p[0]
            ex["fim_type"] = p[1]
            new_ds.append(ex)

    dataset = datasets.Dataset.from_pandas(pd.DataFrame(new_ds))
    return dataset

def fim_remove_types(
    ts_prog : str,
    query_str :str, 
    language: str = "typescript"
    ) -> List[Tuple[str, str]]:
    """
    Make fim prompts for each type annotation in the program, remove all other type annotations
    TODO: inefficient
    """
    parser = lang_to_parser[language]
    lang = lang_to_builder[language]
    
    original = ts_prog
    tree = parser.parse(bytes( ts_prog, "utf-8"))
    query = lang.query(query_str)
    
    captures = query.captures(tree.root_node)
    if len(captures) == 0:
        return []
    captures = merge_captures(captures[::-1]) # walk backwards to preserve idx
    
    prompts = []
    
    ts_prog = tree.text
    for i in range(len(captures)):
        c = captures[i][0]
        captured_type = c.text.decode("utf-8")[1:].strip()
        
        stripped = ts_prog
        for j in range(len(captures)):
            if i != j:
                stripped = replace_between_bytes(stripped, captures[j][0].start_byte, captures[j][0].end_byte, "")
            else:
                stripped = replace_between_bytes(stripped, captures[j][0].start_byte, captures[j][0].end_byte, ": <FILL>")
            
        try:
            prompts.append((stripped.decode("utf-8").strip(), captured_type))
        except:
            prompts.append(("_error_", "_error_"))
        
    return prompts


def merge_captures(captures : list) -> list:
    """
    Type annotations can be nested, i.e. function types. Find nested captures and delete
    the inner ones until only the outermost type annotation is left
    """
    new_captures = []
    # work in pairs, if next capture is a superset of the current one, skip it
    for (curr, nxt) in zip(captures, captures[1:]):
        if (curr[0].start_point[0] == nxt[0].start_point[0] and 
            curr[0].start_point[1] >= nxt[0].start_point[1] and
            curr[0].end_point[0] == nxt[0].end_point[0] and
            curr[0].end_point[1] <= nxt[0].end_point[1]):
            continue
        else:
            new_captures.append(curr)
            
    # add the last capture in all cases
    new_captures.append(captures[-1])       
    return new_captures

def py_remove_annotations(
    program:str,
    placeholder:str,
    ) -> str:
    """
    Remove all caotures from a program except for the FIM type.
    Tree-sitter doesn't capture the colon between parameter and type_annotation, so we have to do it manually
    For each typed parameter, find index of colon and remove everything until parameter end
    """
    lang = lang_to_builder["py"]
    parser = lang_to_parser["py"]
    query = lang.query("((typed_parameter) @tp)")
    tree = parser.parse(bytes(program, "utf8"))
    captures = query.captures(tree.root_node)[::-1]
    program = tree.text
    for c in captures:
        if placeholder in c[0].text.decode("utf-8").strip():
            continue
        colon_idx = find_between_bytes(program, c[0].start_byte, c[0].end_byte, ":")
        if colon_idx == -1:
            raise ValueError("Colon not found")
        program = replace_between_bytes(program, colon_idx, c[0].end_byte, "")
    return program.decode("utf-8")

