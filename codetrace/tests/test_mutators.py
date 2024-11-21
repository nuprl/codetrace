import itertools as it
from codetrace.scripts.typecheck_ds import typecheck_batch
from codetrace.fast_utils import batched_apply, make_batches
from codetrace.py_mutator import (
    PyMutator,
    PY_TYPE_ANNOTATIONS_QUERY,
    RETURN_TYPES as PY_RETURN_TYPES,
    PY_IDENTIFIER_QUERY,
    IMPORT_STATEMENT_QUERY,
    DummyTreeSitterNode
)
from codetrace.ts_mutator import (
    TsMutator,
    TS_IDENTIFIER_QUERY,
    TS_PARAM_TYPES_QUERY,
    TS_TYPE_ANNOTATIONS_QUERY,
    TS_VARIABLE_DECLARATION_QUERY
)
from codetrace.base_mutator import Mutation, TreeSitterLocation,MutationFn
import os
from typing import Union,Tuple,List,Dict
from codetrace.parsing_utils import get_captures

CWD = os.path.dirname(os.path.abspath(__file__))
PROG = os.path.join(CWD, "test_programs")

def get_import_statements(program: str) ->bytes:
    return (b"\n".join([imp.text for imp in
        get_captures(program, IMPORT_STATEMENT_QUERY, "py", "import_statement")]))

def read(path: str) -> str:
    with open(path, "r") as fp:
        return fp.read()

def byte_encode(s: str, encoding="utf-8") -> bytes:
    return bytes(s, encoding)

def read_bytes(path: str) -> bytes:
    return byte_encode(read(path))

def test_extract_type():
    mutator = PyMutator()
    program = """def op(time: __typ2):
	pass"""
    cap = get_captures(program, PY_TYPE_ANNOTATIONS_QUERY, "py","annotation")
    output = mutator.extract_type_from_annotation(cap[0])
    assert output
    assert output.text == b"__typ2"

def test_need_alias():
    mutator = PyMutator()
    program = f"""
from typing import Optional, Container, Callable

from mypy.types import (
    Type, TypeVisitor, UnboundType, AnyType, NoneTyp, TypeVarId, Instance, TypeVarType,
    CallableType, TupleType, TypedDictType, UnionType, Overloaded, ErasedType, PartialType,
    DeletedType, TypeTranslator, UninhabitedType, TypeType, TypeOfAny, LiteralType,
)

class EraseTypeVisitor(TypeVisitor[Type]):

    def visit_unbound_type(self, t: UnboundType) -> {mutator.tree_sitter_placeholder}:
        # TODO: replace with an assert after UnboundType can't leak from semantic analysis.
        return AnyType(TypeOfAny.from_error)
"""
    cap = [mutator.shift_capture_from_char(c, b":",1) for c 
           in get_captures(program, PY_TYPE_ANNOTATIONS_QUERY,"py","annotation")
           if b"UnboundType" in c.text]
    assert len(cap) == 1
    cap = cap[0]
    imps = get_import_statements(program)
    output = mutator.needs_alias(cap, import_statements=imps)
    print(cap.text, imps)
    assert output

    program = f"""
from typing import Optional, Container, Callable

from mypy.types import (
    Type, TypeVisitor, UnboundType, AnyType, NoneTyp, TypeVarId, Instance, TypeVarType,
    CallableType, TupleType, TypedDictType, UnionType, Overloaded, ErasedType, PartialType,
    DeletedType, TypeTranslator, UninhabitedType, TypeType, TypeOfAny, LiteralType,
)

class EraseTypeVisitor(TypeVisitor[Type]):

    def visit_unbound_type(self, t: __tmp) -> {mutator.tree_sitter_placeholder}:
        # TODO: replace with an assert after UnboundType can't leak from semantic analysis.
        return AnyType(TypeOfAny.from_error)
"""

    expected = f"""from typing import TypeAlias
__tmp : TypeAlias = "UnboundType"

from typing import Optional, Container, Callable

from mypy.types import (
    Type, TypeVisitor, UnboundType, AnyType, NoneTyp, TypeVarId, Instance, TypeVarType,
    CallableType, TupleType, TypedDictType, UnionType, Overloaded, ErasedType, PartialType,
    DeletedType, TypeTranslator, UninhabitedType, TypeType, TypeOfAny, LiteralType,
)

class EraseTypeVisitor(TypeVisitor[Type]):

    def visit_unbound_type(self, t: __tmp) -> {mutator.tree_sitter_placeholder}:
        # TODO: replace with an assert after UnboundType can't leak from semantic analysis.
        return AnyType(TypeOfAny.from_error)
"""
    output = mutator.add_aliases_to_program(bytes(program, "utf-8"), [b"__tmp : TypeAlias = \"UnboundType\""])
    assert output == bytes(expected,"utf-8")

    program = """from flask import g, url_for

from byceps.blueprints.site.page.templating import url_for_site_page
from byceps.services.site_navigation import site_navigation_service
from byceps.services.site_navigation.models import (
    NavItem,
    NavItemForRendering,
    NavItemTargetType,
    NavMenuID,
)
from byceps.util.framework.blueprint import create_blueprint
from byceps.util.l10n import get_locale_str

def _to_items_for_rendering(
    site_id: NavItem, items: list[NavItem]
) -> list[NavItemForRendering]:
    return [_to_item_for_rendering(site_id, item) for item in items]"""

    cap = [mutator.shift_capture_from_char(c, b":",1) for c 
           in get_captures(program, PY_TYPE_ANNOTATIONS_QUERY,"py","annotation")
           if b": NavItem" in c.text]
    assert len(cap) == 1
    cap = cap[0]
    imps = get_import_statements(program)
    output = mutator.needs_alias(cap, import_statements=imps)
    print(cap.text, imps)
    assert output

    program = "def func(s: str) -> str:\n\tpass"
    cap = [mutator.shift_capture_from_char(c, b":",2) for c in get_captures(program, PY_TYPE_ANNOTATIONS_QUERY,"py","annotation")
           if b": str" in c.text]
    assert len(cap) == 1
    cap = cap[0]
    imps = get_import_statements(program)
    output = mutator.needs_alias(cap, import_statements=imps)
    print(cap.text, imps)
    assert output


def test_py_add_aliases():
    mutator = PyMutator()
    code = read(f"{PROG}/before_add_alias.py")
    code = mutator.replace_placeholder(code)
    code_bytes = byte_encode(code)
    output = mutator.add_aliases_to_program(
        code_bytes, [
            b'__tmp0 : TypeAlias = "TestUserRedirectView"', 
            b'__tmp1 : TypeAlias = "settings.AUTH_USER_MODEL"',
            b'__tmp2 : TypeAlias = "RequestFactory"'],
        sort=True
    )
    expected = byte_encode(
        mutator.replace_placeholder(
        read(f"{PROG}/after_add_alias.py")
    ))
    with open(f"/tmp/actual_add_alias.py","w") as fp:
        fp.write(output.decode("utf-8"))
    assert output == expected

def test_py_postproc_annotation():
    mutator = PyMutator()
    node = get_captures("def palindrome(s : List[int]):\n\tpass", PY_TYPE_ANNOTATIONS_QUERY, "py", 
                        "annotation")
    assert len(node) == 1
    node = node[0]
    full_annotation = mutator.shift_capture_from_char(node, target_char=b":", shift_amt=0)
    type_only = mutator.shift_capture_from_char(node, target_char=b":", shift_amt=1)
    assert type_only.text == b" List[int]"
    assert full_annotation.text == b": List[int]"

def test_py_postproc_return_types():
    mutator = PyMutator()
    prog = """
def palindrome(s : List[int], **kwargs) -> Union[List[Request], Dict]:
    pass
"""
    program = bytes(prog, "utf-8")
    node = get_captures(program, PY_RETURN_TYPES, "py", "id")
    assert len(node) == 1
    node = node[0]
    assert node.text == b"Union[List[Request], Dict]"
    output = mutator.postprocess_return_type(node, program)
    assert output.text == b"-> Union[List[Request], Dict]"

def test_py_rename_vars():
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

def test_py_rename_types():
    mutator = PyMutator()

    # testing rename types
    program = mutator.replace_placeholder(read(f"{PROG}/before_type_rename.py"))
    type_captures = get_captures(program, PY_TYPE_ANNOTATIONS_QUERY, "py", "annotation")
    type_captures = [mutator.extract_type_from_annotation(c) 
                     for c in type_captures if b"Derp" in c.text or b"UserProfile" in c.text]
    assert len(type_captures) >= 2
    _,type_all_captures,_ = mutator.find_all_other_locations_of_captures(
        program, "int", [], type_captures, []
    )
    assert len(type_all_captures) >= 2
    import_statements = get_import_statements(program)
    output, muts = mutator.mutate_captures(
        program,
        [mutator.rename_types],
        var_rename_captures=[],
        type_rename_captures=type_all_captures,
        remove_captures=[],
        import_statements=import_statements,
        sort=True
    )
    assert any([mutator.needs_alias(c,import_statements=import_statements) for c in type_all_captures])
    assert output
    
    expected = read(f"{PROG}/after_type_rename.py")

    expected = read(f"{PROG}/after_type_rename.py")
    with open(f"/tmp/actual_type_rename.py","w") as fp:
        fp.write(output)
    expected_switched_names = expected.replace("__typ0","__typ-placeholder").replace(
        "__typ1","__typ0").replace("__typ-placeholder","__typ1")
    assert output == expected or output == expected_switched_names

    _,output,_ = mutator.find_all_other_locations_of_captures(
        program, "Derp", [], type_all_captures, [])
    # cannot rename fim
    assert not b"Derp" in [o.text for o in output]

def test_py_delete():
    mutator = PyMutator()
    # testing rename vars
    program = mutator.replace_placeholder(read(f"{PROG}/before_delete.py"))
    del_captures = get_captures(program, PY_TYPE_ANNOTATIONS_QUERY, "py", "annotation")
    del_captures = [mutator.shift_capture_from_char(c, b":", 0) 
                     for c in del_captures if b": str" in c.text]
    assert len(del_captures) > 0
    _,_,delete_captures = mutator.find_all_other_locations_of_captures(
        program, "Realm", [], [], del_captures
    )
    output, _ = mutator.mutate_captures(
        program,
        [mutator.delete_annotations],
        var_rename_captures=[],
        type_rename_captures=[],
        remove_captures=delete_captures
    )
    assert len(delete_captures) > 1
    assert output
    
    expected = read(f"{PROG}/after_delete.py")
    assert output == expected

def test_py_all_muts():
    mutator = PyMutator()
    program = read(f"{PROG}/before_all_muts.py")
    program = mutator.replace_placeholder(program)
    import_statements = get_import_statements(program)

    # delete all UserProfile
    rename_type_muts = [
            mutator.extract_type_from_annotation(n) for n in
            get_captures(program, PY_TYPE_ANNOTATIONS_QUERY, "py", "annotation") if
             b": Addressee" in n.text
        ] + [
            mutator.extract_type_from_annotation(n) for n in
            get_captures(program, PY_TYPE_ANNOTATIONS_QUERY, "py", "annotation") if
             b": str" in n.text
        ]
    
    mutations = mutator.find_all_other_locations_of_captures(
        program,
        "Realm",
        # rename msg_type
        [
            n for n in
            get_captures(program, "((identifier) @id)", "py", "id") if
             b"msg_type" in n.text
        ],
        rename_type_muts,
        [
            mutator.shift_capture_from_char(n, b":", 0) for n in
            get_captures(program, PY_TYPE_ANNOTATIONS_QUERY, "py", "annotation") if
             b": UserProfile" in n.text
        ]
    )
    output,_ = mutator.mutate_captures(
        program, [mutator.rename_types, mutator.rename_vars, mutator.delete_annotations], *mutations,
        import_statements=import_statements
    )
    # output = mutator.revert_placeholder(output)
    expected = read(f"{PROG}/after_all_muts.py")
    with open("/tmp/actual_all_muts.py","w") as fp:
        fp.write(output)
    expected_switched = expected.replace("__typ0", "__typ-placeholder").replace(
        "__typ1","__typ0").replace("__typ-placeholder","__typ1")
    assert output == expected or output == expected_switched

def test_merge_nested_mutations():
    mutator = PyMutator()
    prompt = """def for_stream(
    stream_name: str, 
    topic: str
) -> 'Addressee':"""
    nodes = get_captures(prompt, "((typed_parameter) @s) ((identifier) @s)", "py","s")
    mutations = [
        Mutation(TreeSitterLocation(n), None, None, None) for n in nodes
    ]
    expected = [
        Mutation(TreeSitterLocation(n), None, None, None) for n in nodes
        if n.type == "typed_parameter" or n.text == b"for_stream"
    ]
    output = mutator.merge_nested_mutation(mutations)
    assert set([str(o) for o in output]) == set([str(e) for e in expected])

def test_ts_rename_vars():
    mutator = TsMutator()

    # testing rename vars
    program = mutator.replace_placeholder(read(f"{PROG}/before_var_rename.ts"))
    var_captures = get_captures(program, TS_VARIABLE_DECLARATION_QUERY, "ts", "name")
    assert len(var_captures) > 0
    var_captures = [v for v in var_captures if b"current" in v.text]
    assert len(var_captures) > 0
    var_all_captures,_,_ = mutator.find_all_other_locations_of_captures(
        program, "T", var_captures, [], []
    )
    assert len(var_all_captures) > 0
    output, _ = mutator.mutate_captures(
        program,
        [mutator.rename_vars],
        var_rename_captures=var_all_captures,
        type_rename_captures=[],
        remove_captures=[]
    )
    assert output
    expected = read(f"{PROG}/after_var_rename.ts")
    assert output == expected

def test_ts_rename_types():
    mutator = TsMutator()
    # testing rename types
    program = mutator.replace_placeholder(read(f"{PROG}/before_type_rename.ts"))
    type_captures = get_captures(program, TS_TYPE_ANNOTATIONS_QUERY, "ts", "name")
    type_captures = [mutator.extract_type_from_annotation(c)
            for c in type_captures if b"User" in c.text or b"string" in c.text]
    assert len(type_captures) >= 2
    _,type_all_captures,_ = mutator.find_all_other_locations_of_captures(
        program, "number", [], type_captures, []
    )
    assert len(type_all_captures) >= 2
    output, muts = mutator.mutate_captures(
        program,
        [mutator.rename_types],
        var_rename_captures=[],
        type_rename_captures=type_all_captures,
        remove_captures=[]
    )
    assert output
    
    expected = read(f"{PROG}/after_type_rename.ts")

    expected = read(f"{PROG}/after_type_rename.ts")
    with open(f"/tmp/actual_type_rename.ts","w") as fp:
        fp.write(output)
    expected_switched_names = expected.replace("__typ0","__typ-placeholder").replace(
        "__typ1","__typ0").replace("__typ-placeholder","__typ1")
    assert output == expected or output == expected_switched_names

    # cannot rename fim
    _,output,_ = mutator.find_all_other_locations_of_captures(
        program, "string", [], type_all_captures, [])
    assert not b"string" in [o.text for o in output]

def test_ts_extract_type_annotation():
    mutator = TsMutator()
    program = """
function doesNotReject(block: (() => Promise<any>) | Promise<any>, message?: string | Error): Promise<void>;
    """
    node = get_captures(program, TS_TYPE_ANNOTATIONS_QUERY, "ts","name")
    assert len(node) == 3
    n = [i for i in node if b"(() => Promise<any>) | Promise<any>" in i.text]
    assert len(n) == 1
    n = n[0]
    output = mutator.extract_type_from_annotation(n)
    expected = b"(() => Promise<any>) | Promise<any>"
    assert output.text == expected
    n = [i for i in node if b"string | Error" in i.text]
    assert len(n) == 1
    n = n[0]
    output = mutator.extract_type_from_annotation(n)
    expected = b"string | Error"
    assert output.text == expected

def test_ts_all_muts():
    mutator = TsMutator()
    program = read(f"{PROG}/before_all_muts.ts")
    program = mutator.replace_placeholder(program)

    """
    block -> tmp0
    throws -> tmp1
    delete any
    Error -> typ0
    CallTracker -> typ1
    """
    rename_var_muts = [
            n for n in
            get_captures(program, TS_VARIABLE_DECLARATION_QUERY, "ts", "name") if
             b"block" in n.text
        ] +[
            n for n in
            get_captures(program, TS_VARIABLE_DECLARATION_QUERY, "ts", "name") if
             b"throws" in n.text
        ]
    rename_var_muts = [x for x in rename_var_muts if x]
    
    delete_muts = [
            n for n in get_captures(program, TS_PARAM_TYPES_QUERY, "ts", "name") if
            mutator.extract_type_from_annotation(n).text == b"any"
        ]
    assert len(rename_var_muts) >= 2
    assert len(delete_muts) > 0
    assert any([b"throws" in n.text for n in rename_var_muts])
    
    rename_type_muts = [
            mutator.extract_type_from_annotation(n) for n in
            get_captures(program, TS_TYPE_ANNOTATIONS_QUERY, "ts", "name") if
             b"Error" in n.text
        ] + [
            mutator.extract_type_from_annotation(n) for n in
            get_captures(program, TS_TYPE_ANNOTATIONS_QUERY, "ts", "name") if
             b"CallTracker" in n.text
        ]
    
    rename_type_muts = [x for x in rename_type_muts if x]
    assert any([b"Error" in n.text for n in rename_type_muts])
    assert len(rename_type_muts) >= 2

    mutations = mutator.find_all_other_locations_of_captures(
        program,
        "any",
        rename_var_muts,
        rename_type_muts,
        delete_muts
    )
    assert any([b"block" in n.text for n in mutations[0]])
    assert any([b"CallTracker" in n.text for n in mutations[1]])
    assert any([b"throws" in n.text for n in mutations[0]])
    assert any([b"Error" in n.text for n in mutations[1]])
    assert len(mutations[2]) > 0
    
    output,_ = mutator.mutate_captures(
        program, [mutator.rename_types, mutator.rename_vars, mutator.delete_annotations], *mutations
    )
    assert output
    expected = read(f"{PROG}/after_all_muts.ts")
    with open("/tmp/actual_all_muts.ts","w") as fp:
        fp.write(output)
    
    expected_switched_typ = expected.replace("__typ0", "__typ-placeholder").replace(
        "__typ1","__typ0").replace("__typ-placeholder","__typ1")
    expected_switched_vars = expected.replace("__tmp0", "__tmp-placeholder").replace(
        "__tmp1","__tmp0").replace("__tmp-placeholder","__tmp1")
    expected_switched_vars_and_types = expected.replace("__tmp0", "__tmp-placeholder").replace(
        "__tmp1","__tmp0").replace("__tmp-placeholder","__tmp1").replace(
        "__typ0", "__typ-placeholder").replace(
        "__typ1","__typ0").replace("__typ-placeholder","__typ1")
    assert output == expected or output == expected_switched_vars or \
        output == expected_switched_typ or output == expected_switched_vars_and_types


if __name__ == "__main__":
    import pytest
    pytest.main([os.path.abspath(__file__), "-vv"])