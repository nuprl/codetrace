import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils import *
import datasets
import pandas as pd
import json
from ast import literal_eval
from typing import Tuple
from build_dataset import *
    
# def test_no_type_annotations():
#     """
#     check if there are no type annotations in the dataset.
#     NOTE: weak check because tree-sitter might not parse all programs with Error
#     """
#     dataset = datasets.load_dataset("franlucc/stenotype-eval-type-stripped", split="train")
#     for i,ex in enumerate(dataset):
#         ts_prog = ex["content_type_removed"]
#         tree = TS_PARSER.parse(bytes( ts_prog, "utf8"))
#         query = TS_LANGUAGE.query("((type_annotation) @name)")
#         captures = query.captures(tree.root_node)
#         if len(captures) > 0 not in [49,66,109,213,247,293]: # these are known to have type annotations, weird cases:
#             assert False, f"Found type annotations at index {i}"
            
# def test_no_func_type_annotations():
#     """
#     check if there are no type annotations in the dataset.
#     NOTE: weak check because tree-sitter might not parse all programs with Error
#     """
#     dataset = datasets.load_dataset("franlucc/stenotype-eval-func-type-stripped-v1", split="train")
#     for i,ex in enumerate(dataset):
#         ts_prog = ex["content_type_removed"]
#         tree = TS_PARSER.parse(bytes( ts_prog, "utf8"))
#         query = TS_LANGUAGE.query(QUERY_FUNC_TYPES)
#         captures = query.captures(tree.root_node)
#         if len(captures) > 0:
#             assert False, f"Found type annotations at index {i}"
        

def test_remove_types():
    ts_prog = "function foo(a: number, b: string): number {\n\treturn 1;\n}"
    new_prog, types = remove_types(ts_prog)
    new_prog_funcs, types_funcs = remove_types(ts_prog, query_str=QUERY_FUNC_TYPES)
    gold = "function foo(a, b) {\n\treturn 1;\n}"
    
    assert new_prog_funcs == new_prog == gold, f"===OLD===:\n{ts_prog}\n===NEW===:\n{new_prog}"
    assert types == types_funcs == {((0, 34), (0, 42)): "number", 
                     ((0, 25), (0, 33)): "string", 
                     ((0, 14), (0, 22)): "number"}, types
    
    new_prog, types = remove_types_with_idx("\n".join([ts_prog]*10), QUERY_FUNC_TYPES)
    gold = "\n".join([gold]*10)
    assert new_prog == gold, f"===OLD===:\n{gold}\n===NEW===:\n{new_prog}"
    

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

    
def test_remove_types_full_idx():
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
    gold_prog= """
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
    gold_prog_fn= """
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
    new_prog, _ = remove_types_with_idx(ts_prog)
    func_new_prog, _ = remove_types_with_idx(ts_prog, query_str=QUERY_FUNC_TYPES)
    assert new_prog == gold_prog, f"===OLD===:\n {gold_prog}\n===NEW===:\n{new_prog}"
    assert func_new_prog == gold_prog_fn, f"===OLD===:\n {gold_prog_fn}\n===NEW===:\n{func_new_prog}"
    

def test_byte_filter():
    prog= """
// 周期-判定星期几
function checkWeek(year: number, month: number, day: number) {
    let weekday = [
        "星期日",
        "星期一",
        "星期二",
        "星期三",
        "星期四",
        "星期五",
        "星期六",
    ];
    let newDate = new Date(`${year}/${month}/${day}`);
    let mark = newDate.getDay();

    return {
        value: mark === 0 ? 7 : mark,
        label: weekday[mark],
    };
}
"""
    gold = """
// 周期-判定星期几
function checkWeek(year, month, day) {
    let weekday = [
        "星期日",
        "星期一",
        "星期二",
        "星期三",
        "星期四",
        "星期五",
        "星期六",
    ];
    let newDate = new Date(`${year}/${month}/${day}`);
    let mark = newDate.getDay();

    return {
        value: mark === 0 ? 7 : mark,
        label: weekday[mark],
    };
}
"""
    new_prog, _ = remove_types_with_idx(prog)
    assert new_prog == gold, f"===OLD===:\n {gold}\n===NEW===:\n{new_prog}"
    
if __name__ == "__main__":
    test_remove_types()
    test_remove_types_multiline()
    test_remove_types_full()
    test_remove_types_full_idx()
    test_byte_filter()
    # test_no_func_type_annotations()
    