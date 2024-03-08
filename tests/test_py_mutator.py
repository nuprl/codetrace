from codetrace.py_mutator import (
    rename_var,
    TYPED_IDENTIFIERS,
    EditableNode,
    edit_nodes,
    remove_type_annotion,
    random_mutations,
)
import pytest
from codetrace.utils import PY_LANGUAGE, PY_PARSER


def test_rename_class_longer():
    input_buffer = b"""
    class Foo:
        pass
    
    x = Foo()
    def f(y: Foo):
        pass
    """
    input_tree = PY_PARSER.parse(input_buffer)
    output_buffer, _ = rename_var(input_buffer, input_tree.root_node, b"Foo", b"Barr")
    assert (
        output_buffer
        == b"""
    class Barr:
        pass
    
    x = Barr()
    def f(y: Barr):
        pass
    """
    )


def test_rename_var_longer():
    input_buffer = b"""
    def foo(x):
        return x + 1
    """
    input_tree = PY_PARSER.parse(input_buffer)
    output_buffer, _ = rename_var(input_buffer, input_tree.root_node, b"x", b"yyyy")
    assert (
        output_buffer
        == b"""
    def foo(yyyy):
        return yyyy + 1
    """
    )


def test_rename_var_shorter():
    input_buffer = b"""
    def foo(yyyy):
        return yyyy + 1
    """
    input_tree = PY_PARSER.parse(input_buffer)
    output_buffer, _ = rename_var(input_buffer, input_tree.root_node, b"yyyy", b"x")
    assert (
        output_buffer
        == b"""
    def foo(x):
        return x + 1
    """
    )


def test_rename_then_remove_type():
    input_buffer = b"""
    def foo(yyyy, z: int):
        return yyyy + 1
    """

    input_tree = PY_PARSER.parse(input_buffer)
    all_type_annotations = [
        EditableNode.from_node(node)
        for (node, _) in TYPED_IDENTIFIERS.captures(input_tree.root_node)
    ]
    output_buffer, edits = rename_var(input_buffer, input_tree.root_node, b"yyyy", b"x")
    edit_nodes(edits, all_type_annotations)
    output_buffer, _ = remove_type_annotion(output_buffer, all_type_annotations[0])
    assert (
        output_buffer
        == b"""
    def foo(x, z):
        return x + 1
    """
    )


def test_overlapped_rename_then_remove_type_longer():
    input_buffer = b"""
    def foo(z: int):
        return z + 1
    """

    input_tree = PY_PARSER.parse(input_buffer)
    all_type_annotations = [
        EditableNode.from_node(node)
        for (node, _) in TYPED_IDENTIFIERS.captures(input_tree.root_node)
    ]
    output_buffer, edits = rename_var(input_buffer, input_tree.root_node, b"z", b"zzz")
    edit_nodes(edits, all_type_annotations)
    output_buffer, _ = remove_type_annotion(output_buffer, all_type_annotations[0])
    assert (
        output_buffer
        == b"""
    def foo(zzz):
        return zzz + 1
    """
    )


def test_overlapped_rename_then_remove_type_shorter():
    input_buffer = b"""
    def foo(zzz: int):
        return zzz + 1
    """

    input_tree = PY_PARSER.parse(input_buffer)
    all_type_annotations = [
        EditableNode.from_node(node)
        for (node, _) in TYPED_IDENTIFIERS.captures(input_tree.root_node)
    ]
    output_buffer, edits = rename_var(input_buffer, input_tree.root_node, b"zzz", b"z")
    edit_nodes(edits, all_type_annotations)
    output_buffer, _ = remove_type_annotion(output_buffer, all_type_annotations[0])
    assert (
        output_buffer
        == b"""
    def foo(z):
        return z + 1
    """
    )


def test_random_mutations():
    # Two bound variables and one type annotation. So, three possible mutation points.
    input_code = """
    def foo(xxxxxxxxx, z: int):
        return xxxxxxxxx + y + 1
    """
    all_mutations = list(random_mutations(input_code, -1))
    assert len(all_mutations) == 3


def test_random_mutations_with_fixed_annotation():
    # Two bound variables and one type annotations. So, three possible mutation points.
    # We also hold the type annotation bool fixed.
    input_code = """
    def foo(xxxxxxxxx: bool, z: int):
        return xxxxxxxxx + y + 1
    """.lstrip()
    # Notice that the location can be inexact, just has to be interior.
    all_mutations = list(random_mutations(input_code, len("def foo(xxxxxx ")))
    assert len(all_mutations) == 3



def test_target_index_match_shorter():
    input_code = """
    def foo(zzzzzzzzzz: int):
        return zzzzzzzzzz + 1
    """

    target_type_index = input_code.find(" int")
    all_mutations = list(random_mutations(input_code, target_type_index))
    assert len(all_mutations) == 1
    (new_target_type_index, output_code) = all_mutations[0]
    assert new_target_type_index < target_type_index
    assert output_code[new_target_type_index : new_target_type_index + 4] == " int"

def test_target_index_match_longer():
    input_code = """
    def foo(x: int):
        return x + 1
    """

    target_type_index = input_code.find(" int")
    all_mutations = list(random_mutations(input_code, target_type_index))
    assert len(all_mutations) == 1
    (new_target_type_index, output_code) = all_mutations[0]
    assert new_target_type_index > target_type_index
    assert output_code[new_target_type_index : new_target_type_index + 4] == " int"