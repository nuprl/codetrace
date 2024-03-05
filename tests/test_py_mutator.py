from codetrace.py_mutator import rename_var, get_bound_vars, rename_var_with_index
import pytest

def test_rename_with_index():
    (new_target_index, new_code) = rename_var_with_index(
        """
        def foo(x):
            return x + 1
        """,
        20,
        "x",
        "xxx",
    )
    assert new_target_index == 20 + 2



def test_rename_with_index():
    (new_target_index, new_code) = rename_var_with_index(
        """
        def foo(x):
            return x + 1
        """,
        40,
        "x",
        "xxx",
    )
    assert new_target_index == 40 + 4


def test_rename_same_imported_name():
    assert (
        rename_var(
            """
        import x
        from x import *
        from .x import x

        class x:
            pass

        def foo(x):
            return x + 1
        """,
            "x",
            "y",
        )
        == """
        import x
        from x import *
        from .x import x

        class x:
            pass

        def foo(y):
            return y + 1
        """
    )


def test_rename_mixin():
    assert (
        rename_var(
            """
        class C(x):
            pass
        """,
            "x",
            "y",
        )
        == """
        class C(x):
            pass
        """
    )


def test_rename_method_arg():
    assert (
        rename_var(
            """
        class C:
            def foo(x):            
                pass
        """,
            "x",
            "y",
        )
        == """
        class C:
            def foo(y):            
                pass
        """
    )


def test_rename_method_body():
    assert (
        rename_var(
            """
        class C:
            def foo():
                x = 10
        """,
            "x",
            "y",
        )
        == """
        class C:
            def foo():
                y = 10
        """
    )

@pytest.mark.xfail
def test_get_bound_vars_class_ref():
    """
    It is not safe to rename a identifier that is used as a class name. This
    requires some sophistication.
    """
    assert (
        get_bound_vars(
            """
        class C(x):
        
        def foo(x):
            return x + 1
        """
        )
        == {}
    )


def test_rename_var_1():
    assert (
        rename_var(
            """
        def foo(x):
            return x + 1
        """,
            "x",
            "y",
        )
        == """
        def foo(y):
            return y + 1
        """
    )
