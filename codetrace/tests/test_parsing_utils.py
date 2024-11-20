from codetrace.parsing_utils import (
    get_captures,
    STARCODER_FIM,
    CODELLAMA_FIM_CHAT,
    is_in_capture_range,
    find_between_bytes,
    replace_between_bytes,
)
def test_replace_between_bytes():
    text = b"I ponder when I will be replaced"
    start_byte = text.index(b"when I w")
    end_byte = text.index(b"ill")
    assert text.index(b"w") == start_byte
    assert text.index(b"i") == end_byte
    replacement = b"REPLACEMENT"
    expected = b"I ponder REPLACEMENTill be replaced"
    output = replace_between_bytes(text, start_byte, end_byte, replacement)
    assert output == expected

    prog = bytes("def is_palindrome():","utf-8")
    capture = get_captures(prog, "(function_definition name: ((identifier) @id))", "py", "id")[0]
    assert capture.text == b"is_palindrome"
    output = replace_between_bytes(prog, capture.start_byte, capture.end_byte, b"func")
    expected = b"""def func():"""
    assert output == expected

def test_find_between_bytes():
    prog = bytes("def is_palinedrome():","utf-8")
    capture = get_captures(prog, "(function_definition name: ((identifier) @id))", "py", "id")[0]
    assert capture.text == b"is_palinedrome"
    output = find_between_bytes(capture.text, capture.start_byte + capture.text.index(b"d"), capture.end_byte, b"e")
    expected = capture.text.rindex(b"e")
    assert output == expected

def test_is_in_capture_range():
    prog = """
import math

def outer_function(x, y):
    def inner_function(a, b):
        return a * b
    result = inner_function(x, y)
    return result + math.sqrt(x)

class MyClass:
    def __init__(self, name):
        self.name = name

    def greet(self):
        print(f"Hello, {self.name}!")

obj = MyClass("Alice")
obj.greet()
"""
    capture = get_captures(prog, "((string) @s)", "py", "s")
    alice_string = [c for c in capture if c.text == b'"Alice"'][0]
    fmt_string = [c for c in capture if b'Hello' in c.text][0]
    class_defs = get_captures(prog, "((class_definition) @s)", "py","s")
    expressions = get_captures(prog, "((expression_statement) @s)","py","s")
    assignment = get_captures(prog, "((assignment) @s)", "py","s")
    assert alice_string.text == b'"Alice"'

    assert is_in_capture_range(alice_string, assignment)
    assert is_in_capture_range(alice_string, expressions)
    assert not is_in_capture_range(alice_string, class_defs)

    assert is_in_capture_range(fmt_string, class_defs)
    assert is_in_capture_range(fmt_string, expressions)
    assert not is_in_capture_range(fmt_string, assignment)

def test_captures():
    prompt = "def func:"
    query="((identifier) @name)"
    captures = get_captures(prompt, query, "python", "name")
    assert len(captures) == 1, captures
    assert captures[0].text.decode("utf-8") == "func", captures

def test_fim():
    prompt = "hi my name is <FILL>! Nice to meet you"
    fim_prompt = STARCODER_FIM.placeholder_to_fim(prompt)
    unfimmed_prompt = STARCODER_FIM.unfim(fim_prompt + "George")
    placeholder_prompt = STARCODER_FIM.fim_to_placeholder(fim_prompt)
    assert placeholder_prompt == "hi my name is <FILL>! Nice to meet you"
    assert unfimmed_prompt == "hi my name is George! Nice to meet you"
    assert fim_prompt == \
    f"{STARCODER_FIM.prefix}hi my name is {STARCODER_FIM.suffix}! Nice to meet you{STARCODER_FIM.middle}"

def test_fimchat():
    program = '''
def is_palindrome(s: <FILL>) -> bool:
    return s[::-1] == s
'''.strip()
    program_prefix = "def is_palindrome(s: "
    result = CODELLAMA_FIM_CHAT.placeholder_to_fim(program)
    expected = [
        {"role": "user", 
         "content": f"Continue this program with the correct substitution for <FILL>:\n\n{program}"},
        {"role": "assistant", "content": program_prefix}
    ]
    assert result == expected
    assert CODELLAMA_FIM_CHAT.chat_template() != expected

if __name__ == "__main__":
    import pytest
    import os
    pytest.main([os.path.abspath(__file__), "-vv"])