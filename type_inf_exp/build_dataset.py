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

QUERY_ALL_TYPES = """((type_annotation) @name)"""
QUERY_FUNC_TYPES = """
(required_parameter
      pattern: (_) (type_annotation) @tp)
  (optional_parameter
      pattern: (_) (type_annotation) @tp)
    return_type: (type_annotation) @tp
"""

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
    

def remove_types(ts_prog : str, query_str :str = QUERY_ALL_TYPES) -> Tuple[str, dict]:
    """
    remove all type annotations from the program
    
    NOTE: to re-insert types from type map, start at idx 0 and insert types in order
    so index gets updated correctly
    """
    tree = TS_PARSER.parse(bytes( ts_prog, "utf8"))
    query = TS_LANGUAGE.query(query_str)
    
    captures = query.captures(tree.root_node)[::-1]
    if len(captures) == 0:
        return ts_prog, {}
    captures = merge_captures(captures)

    # end_byte = tree.included_ranges[0].end_byte
    # start_point = tree.included_ranges[0].start_point
    # end_point = tree.included_ranges[0].end_point
    
    type_map = {}

    for c in captures:
        c = c[0]
        captured_type = c.text.decode("utf-8")[1:].strip()
        
        type_map[(c.start_point, c.end_point)] = captured_type

        ts_prog = remove_between_points(ts_prog, c.start_point, c.end_point)
    return ts_prog, type_map


def filter_types(dataset : datasets.Dataset, query_str : str = QUERY_ALL_TYPES) -> datasets.Dataset:
    """
    remove all type annotations from the dataset
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
    
"""
TESTS
"""

def test_no_type_annotations():
    """
    check if there are no type annotations in the dataset.
    NOTE: weak check because tree-sitter might not parse all programs with Error
    """
    dataset = datasets.load_dataset("franlucc/stenotype-eval-type-stripped", split="train")
    for i,ex in enumerate(dataset):
        ts_prog = ex["content_type_removed"]
        tree = TS_PARSER.parse(bytes( ts_prog, "utf8"))
        query = TS_LANGUAGE.query("((type_annotation) @name)")
        captures = query.captures(tree.root_node)
        if len(captures) > 0 and i not in [49,66,109,213,247,293]: # these are known to have type annotations, weird cases
            assert False, f"Found type annotations at index {i}"
        

def test_remove_types():
    ts_prog = "function foo(a: number, b: string): number {\n\treturn 1;\n}"
    new_prog, types = remove_types(ts_prog)
    new_prog_funcs, types_funcs = remove_types(ts_prog, query_str=QUERY_FUNC_TYPES)
    
    assert new_prog_funcs == new_prog == "function foo(a, b) {\n\treturn 1;\n}", f"===OLD===:\n{ts_prog}\n===NEW===:\n{new_prog}"
    assert types == types_funcs == {((0, 34), (0, 42)): "number", 
                     ((0, 25), (0, 33)): "string", 
                     ((0, 14), (0, 22)): "number"}, types
    

def test_remove_types_multiline():
    ts_prog = """
class Point {
  x: number;
  y: number;
}

function foo(a: number, b: string): number | string {
	if (a > 0) {
        return 1;
    } else {
        return b;
    }
}
function foo(a: number, b: string): number {
    return 1;
}
"""
    ts_prog_new= """
class Point {
  x;
  y;
}

function foo(a, b) {
	if (a > 0) {
        return 1;
    } else {
        return b;
    }
}
function foo(a, b) {
    return 1;
}
"""
    new_prog, types = remove_types(ts_prog)
    assert new_prog == ts_prog_new, f"===OLD===:\n {ts_prog}\n===NEW===:\n{new_prog}"
    
    
def test_remove_types_full():
    ts_prog = """
class Point {
  x: number;
  y: number;
}
 
function foo(a: number, b: string): number | string {
	if (a > 0) {
        return 1;
    } else {
        return b;
    }
}
function greeter(fn: (a: string) => void) {
  fn("Hello, World");
}
"""
    ts_prog_new= """
class Point {
  x;
  y;
}
 
function foo(a, b) {
	if (a > 0) {
        return 1;
    } else {
        return b;
    }
}
function greeter(fn) {
  fn("Hello, World");
}
"""
    ts_prog_new_only_funcs= """
class Point {
  x: number;
  y: number;
}
 
function foo(a, b) {
	if (a > 0) {
        return 1;
    } else {
        return b;
    }
}
function greeter(fn) {
  fn("Hello, World");
}
"""
    new_prog, _ = remove_types(ts_prog)
    func_new_prog, _ = remove_types(ts_prog, query_str=QUERY_FUNC_TYPES)
    assert new_prog == ts_prog_new, f"===OLD===:\n {ts_prog}\n===NEW===:\n{new_prog}"
    assert func_new_prog == ts_prog_new_only_funcs, f"===OLD===:\n {ts_prog}\n===NEW===:\n{func_new_prog}"
    
if __name__ == "__main__":
    test_remove_types()
    test_remove_types_multiline()
    test_remove_types_full()
    # test_no_type_annotations()
    