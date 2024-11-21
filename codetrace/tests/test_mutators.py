from codetrace.py_mutator import (
    PyMutator,
    PY_TYPE_ANNOTATIONS_QUERY,
    RETURN_TYPES as PY_RETURN_TYPES,
    PY_IDENTIFIER_QUERY,
    get_toplevel_parent
)
# from codetrace.py_mutator import (
#     random_mutate as _py_mutate,
#     mutate_captures as py_mutate_captures,
#     find_mutation_locations as py_find_mut_locs
# )
# from codetrace.ts_mutator import (
#     random_mutate as _ts_mutate,
#     mutate_captures as py_mutate_captures,
#     find_mutation_locations as py_find_mut_locs
# )
# merge_nested_mutations
from codetrace.ts_mutator import TsMutator
import os
from codetrace.parsing_utils import get_captures

CWD = os.path.dirname(os.path.abspath(__file__))
PROG = os.path.join(CWD, "test_programs")

def read(path: str) -> str:
    with open(path, "r") as fp:
        return fp.read()

def byte_encode(s: str, encoding="utf-8") -> bytes:
    return bytes(s, encoding)

def read_bytes(path: str) -> bytes:
    return byte_encode(read(path))

def test_py_add_aliases():
    mutator = PyMutator()
    code = read(f"{PROG}/before_add_alias.py")
    code = mutator.replace_placeholder(code)
    code_bytes = byte_encode(code)
    output = mutator.add_type_aliases(
        code_bytes, [
            b'__tmp0 : TypeAlias = "TestUserRedirectView"', 
            b'__tmp1 : TypeAlias = "settings.AUTH_USER_MODEL"',
            b'__tmp2 : TypeAlias = "RequestFactory"']
    )
    expected = byte_encode(
        mutator.replace_placeholder(
        read(f"{PROG}/after_add_alias.py")
    ))
    assert output == expected

def test_py_postproc_annotation():
    mutator = PyMutator()
    node = get_captures("def palindrome(s : List[int]):\n\tpass", PY_TYPE_ANNOTATIONS_QUERY, "py", 
                        "annotation")[0]
    full_annotation = mutator.postprocess_type_annotation(node, target_char=b":", shift_amt=0)
    type_only = mutator.postprocess_type_annotation(node, target_char=b":", shift_amt=1)
    assert type_only.text == b" List[int]"
    assert full_annotation.text == b": List[int]"

def test_py_postproc_return_types():
    mutator = PyMutator()
    prog = """
def palindrome(s : List[int], **kwargs) -> Union[List[Request], Dict]:
    pass
"""
    program = bytes(prog, "utf-8")
    node = get_captures(program, PY_RETURN_TYPES, "py", "id")[0]
    assert node.text == b"Union[List[Request], Dict]"
    output = mutator.postprocess_return_type(node, program)
    assert output.text == b"-> Union[List[Request], Dict]"

def test_py_mutate_captures():
    mutator = PyMutator()

    # testing rename vars
    program = mutator.replace_placeholder(read(f"{PROG}/before_var_rename.py"))
    var_captures = get_captures(program, PY_IDENTIFIER_QUERY, "py", "id")
    var_captures = [v for v in var_captures if b"forces" in v.text]
    var_all_captures,_,_ = mutator.find_all_other_locations_of_captures(
        program, "float", var_captures, [], []
    )
    output, _ = mutator.mutate_captures(
        program,
        [mutator.rename_vars],
        var_rename_captures=var_all_captures,
        type_rename_captures=[],
        remove_captures=[]
    )
    assert len(var_all_captures) == 8
    assert output
    expected = read(f"{PROG}/after_var_rename.py")
    assert output == expected

    # # testing rename types
    # program = mutator.replace_placeholder(read(f"{PROG}/before_type_rename.py"))
    # type_captures = get_captures(program, PY_TYPE_ANNOTATIONS_QUERY, "py", "id")
    # type_captures = []
    # type_all_captures,_,_ = mutator.find_all_other_locations_of_captures(
    #     program, "float", type_captures, [], []
    # )
    # output, _ = mutator.mutate_captures(
    #     program,
    #     [mutator.rename_types],
    #     var_rename_captures=[],
    #     type_rename_captures=type_captures,
    #     remove_captures=[]
    # )
    # assert len(type_all_captures) == 8
    # assert output
    # expected = read(f"{PROG}/after_type_rename.py")
    # assert output == expected

def test_py_needs_alias():
    mutator = PyMutator()
    # import statement does needs alias
    prog = '''
    from math.hulu import (
	call as p,
    s, t
)
'''
    cap = get_captures(prog, "((identifier) @s)", "py", "s")
    cap = [c for c in cap if c.text == b"p"][0]
    assert mutator.needs_alias(cap)

    prog = '''
    from math.hulu.tru import (
	call.terry as p,
    s, t
)
'''
    cap = get_captures(prog, "((identifier) @s)", "py", "s")
    cap = [c for c in cap if c.text == b"p"][0]
    assert mutator.needs_alias(cap)


def test_get_toplevel_parent():
    prog = """
import math
from torch.utils import (
	data.DataLoader as dl
)

def outer_function(x, y):
    import boo
    def inner_function(a, b):
        return a * b
    result = inner_function(x, y)
    return result + math.sqrt(x)
"""
    captures = get_captures(prog, "((identifier) @s)", "py", "s")
    cap = [c for c in captures if c.text == b"dl"]
    assert len(cap) == 1
    cap = cap[0]
    parent = get_toplevel_parent(cap)
    assert parent.type == "import_from_statement"
    assert parent.text == b"""from torch.utils import (
	data.DataLoader as dl
)"""

    cap = [c for c in captures if c.text == b"sqrt"]
    assert len(cap) == 1
    cap = cap[0]
    parent = get_toplevel_parent(cap)
    assert parent.type == "return_statement"
    assert parent.text == b"return result + math.sqrt(x)"

    captures = get_captures(prog, "((identifier) @s)", "py", "s")
    cap = [c for c in captures if c.text == b"boo"]
    assert len(cap) == 1
    cap = cap[0]
    parent = get_toplevel_parent(cap)
    assert parent.type == "import_statement"
    assert parent.text == b"import boo"


if __name__ == "__main__":
    import pytest
    pytest.main([os.path.abspath(__file__), "-vv"])