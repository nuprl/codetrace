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

QUERY_ALL_TYPES = """((type_annotation) @name)"""
QUERY_FUNC_TYPES = """
(required_parameter
      pattern: (_) (type_annotation) @tp)
  (optional_parameter
      pattern: (_) (type_annotation) @tp)
    return_type: (type_annotation) @tp
"""

"""
The following functions remove all type annotations, keeping track of their position
in a bottom-up incremental way.
"""

def filter_types(dataset : datasets.Dataset, query_str : str = QUERY_ALL_TYPES) -> datasets.Dataset:
    """
    remove all type annotations from the dataset. Stores indirect index access
    """
    new_ds = []
    for ex in dataset:
        content, types = remove_types(ex["content"], query_str)
        ex["content_type_removed"] = content
        types = {str(k[0]) + "-" + str(k[1]): v for k, v in types.items()}
        ex["type_map"] = json.dumps(types)
        new_ds.append(ex)

    dataset = datasets.Dataset.from_pandas(pd.DataFrame(new_ds))
    return dataset
    

def remove_types(ts_prog : str, query_str :str = QUERY_ALL_TYPES) -> Tuple[str, dict]:
    """
    remove all type annotations from the program
    
    NOTE: to re-insert types from type map, start at idx 0 and insert types in order
    so index gets updated correctly. Not for direct indexing
    """
    tree = TS_PARSER.parse(bytes( ts_prog, "utf-8"))
    query = TS_LANGUAGE.query(query_str)
    
    captures = query.captures(tree.root_node)[::-1] # walk backwards to preserve idx
    if len(captures) == 0:
        return ts_prog, {}
    captures = merge_captures(captures) 
    
    type_map = {}

    ts_prog = tree.text
    for c in captures:
        c = c[0]
        captured_type = c.text.decode("utf-8")[1:].strip()
        
        type_map[(c.start_point, c.end_point, c.start_byte, c.end_byte)] = captured_type

        ts_prog = replace_between_bytes(ts_prog, c.start_byte, c.end_byte, "")
    return ts_prog.decode("utf-8").strip(), type_map

"""
The following functions remove all type annotations but one <FILL> fim type
"""

def make_typeinf_prompts(dataset : datasets.Dataset, query_str : str = QUERY_ALL_TYPES) -> datasets.Dataset:
    """
    Make a dataset with all type annotations removed and a prompt for each type annotation
    """
    new_ds = []
    for ex in dataset:
        prompts = fim_remove_types(ex["content"], query_str)
        for p in prompts:
            ex = ex.copy()
            ex["fim_program"] = p[0]
            ex["fim_type"] = json.dumps(p[1])
            new_ds.append(ex)

    dataset = datasets.Dataset.from_pandas(pd.DataFrame(new_ds))
    return dataset

def fim_remove_types(ts_prog : str, query_str :str = QUERY_ALL_TYPES) -> List[Tuple[str, str]]:
    """
    Make fim prompts for each type annotation in the program, remove all other type annotations
    """
    original = ts_prog
    tree = TS_PARSER.parse(bytes( ts_prog, "utf-8"))
    query = TS_LANGUAGE.query(query_str)
    
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
            if i < j:
                stripped = replace_between_bytes(stripped, captures[j][0].start_byte, captures[j][0].end_byte, "")
            elif i == j:
                stripped = replace_between_bytes(stripped, captures[j][0].start_byte, captures[j][0].end_byte, "<FILL>")
        
        # # for some reason this is necessary, index won't update correctly otherwise
        # with tempfile.NamedTemporaryFile(mode='w+t', delete=False) as temp:
        #     temp.write(stripped)
        #     temp.seek(0)
        #     stripped = temp.read()
            
        prompts.append((stripped.decode("utf-8").strip(), captured_type))
        # ts_prog = replace_between_bytes(ts_prog, c.start_byte, c.end_point, "")
        
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

