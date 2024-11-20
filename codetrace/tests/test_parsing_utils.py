from codetrace.parsing_utils import (
    get_captures,
    STARCODER_FIM,
    CODELLAMA_FIM_CHAT,
    is_in_capture_range,
    find_between_bytes,
    replace_between_bytes,
)

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
    program_prefix = "def is_palindrome(s: <MID>"
    result = CODELLAMA_FIM_CHAT.placeholder_to_fim(program)
    expected = [
        {"role": "user", 
         "content": f"Continue this program with the correct type annotation after <MID>:\n\n{program}"},
        {"role": "assistant", "content": program_prefix}
    ]
    assert result == expected
    assert CODELLAMA_FIM_CHAT.chat_template() != expected