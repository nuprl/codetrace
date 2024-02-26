from codetrace.type_inf_exp.build_dataset import *

def test_remove_types():
    ts_prog = "function foo(a: number, b: string): number {\n\treturn 1;\n}"
    new_prog, types = remove_types(ts_prog)
    new_prog_funcs, types_funcs = remove_types(ts_prog, query_str=QUERY_FUNC_TYPES)
    gold = "function foo(a, b) {\n\treturn 1;\n}"
    
    assert new_prog_funcs == new_prog == gold, f"===OLD===:\n{ts_prog}\n===NEW===:\n{new_prog}"
    assert types == types_funcs == {((0, 34), (0, 42), 34, 42): "number", 
                     ((0, 25), (0, 33), 25, 33): "string", 
                     ((0, 14), (0, 22), 14, 22): "number"}, types
    
    prompts = fim_remove_types("\n".join([ts_prog]*10), QUERY_FUNC_TYPES)
    gold = "function foo(a, b)<FILL> {\n\treturn 1;\n}"
    assert prompts[0], f"===OLD===:\n{gold}\n===NEW===:\n{new_prog}"
    

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
""".strip()
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
""".strip()
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
""".strip()
    new_prog, _ = remove_types(ts_prog)
    func_new_prog, _ = remove_types(ts_prog, query_str=QUERY_FUNC_TYPES)
    assert new_prog == ts_prog_new, f"===GOLD===:\n {ts_prog_new}\n===NEW===:\n{new_prog}"
    assert func_new_prog == ts_prog_new_only_funcs, f"===GOLD===:\n {ts_prog_new_only_funcs}\n===NEW===:\n{func_new_prog}"

    
if __name__ == "__main__":
    test_remove_types()
    test_remove_types_multiline()
    test_remove_types_full()
    print("All tests passed!")