"""
take stenotype-eval-dataset and remove all type annotations but keep track of them

TODO:
- do I also remove types from declarations? or just function signatures?
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils import *
import datasets
import pandas as pd
import json
from ast import literal_eval
from typing import Tuple

QUERY_ALL_TYPES = """((type_annotation) @name)"""
QUERY_FUNC_TYPES = """
(required_parameter
      pattern: (_) (type_annotation) @tp)
  (optional_parameter
      pattern: (_) (type_annotation) @tp)
    return_type: (type_annotation) @tp
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


def filter_types_with_idx(dataset : datasets.Dataset, query_str : str = QUERY_ALL_TYPES) -> datasets.Dataset:
    """
    remove all type annotations from the dataset. Stores direct index access
    """
    new_ds = []
    for ex in dataset:
        content, types = remove_types_with_idx(ex["content"], query_str)
        ex["content_type_removed"] = content
        ex["type_map"] = json.dumps(types)
        new_ds.append(ex)

    dataset = datasets.Dataset.from_pandas(pd.DataFrame(new_ds))
    return dataset
    

def remove_types(ts_prog : str, query_str :str = QUERY_ALL_TYPES) -> Tuple[tuple, dict]:
    """
    remove all type annotations from the program
    
    NOTE: to re-insert types from type map, start at idx 0 and insert types in order
    so index gets updated correctly. Not for direct indexing
    """
    tree = TS_PARSER.parse(bytes( ts_prog, "utf-8"))
    query = TS_LANGUAGE.query(query_str)
    
    captures = query.captures(tree.root_node)[::-1]
    if len(captures) == 0:
        return ts_prog, {}
    captures = merge_captures(captures)
    
    type_map = {}

    for c in captures:
        c = c[0]
        captured_type = c.text.decode("utf-8")[1:].strip()
        
        type_map[(c.start_point, c.end_point)] = captured_type

        ts_prog = remove_between_points(ts_prog, c.start_point, c.end_point)
    return ts_prog, type_map


def remove_types_with_idx(ts_prog : str, query_str :str = QUERY_ALL_TYPES) -> Tuple[int, dict]:
    """
    remove all type annotations from the program
    
    NOTE: Supports direct indexing to insert _one_ type annotation back into
    the stripped program.
    Only issue is multiline type annotations will remain multiline
    """
    original = ts_prog
    tree = TS_PARSER.parse(bytes( ts_prog, "utf-8"))
    query = TS_LANGUAGE.query(query_str)
    
    captures = query.captures(tree.root_node)
    if len(captures) == 0:
        return ts_prog, {}
    captures = merge_captures(captures[::-1])
    
    type_map = {}

    for i in range(len(captures)):
        c = captures[i][0]
        captured_type = c.text.decode("utf-8")[1:].strip()
        ts_prog = remove_between_points(ts_prog, c.start_point, c.end_point)
        
        stripped = original
        for j in range(len(captures)):
            if i != j:
                stripped = remove_between_points(stripped, captures[j][0].start_point, captures[j][0].end_point)
        # find idx of captured type in stripped program
        idx = stripped.find(captured_type)
        type_map[idx] = captured_type
        
    return ts_prog, type_map


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

