import glob
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import os
import json
from tree_sitter import Language, Parser
import tree_sitter
import tempfile
import re

vendor = "/home/franlucc/llm_libs"
parent = Path(__file__).parent
Language.build_library(
    f"{parent}/build/my-languages.so",
    [f"{vendor}/tree-sitter-typescript/typescript",f"{vendor}/tree-sitter-python"],
)
TS_LANGUAGE = Language(f"{parent}/build/my-languages.so", "typescript")
TS_PARSER = Parser()
TS_PARSER.set_language(TS_LANGUAGE)

PY_LANGUAGE = Language(f"{parent}/build/my-languages.so", "python")
PY_PARSER = Parser()
PY_PARSER.set_language(PY_LANGUAGE)

lang_to_parser = {"typescript" : TS_PARSER, "python" : PY_PARSER, "py" : PY_PARSER, "ts" : TS_PARSER}
lang_to_builder = {"typescript" : TS_LANGUAGE, "python" : PY_LANGUAGE, "py" : PY_LANGUAGE, "ts" : TS_LANGUAGE}

class FimObj:
    def __init__(self,
                 fim_prefix : str,
                 fim_suffix : str,
                 fim_token : str,
                 fim_placeholder : str):
        self.prefix = fim_prefix
        self.suffix = fim_suffix
        self.token = fim_token
        self.placeholder = fim_placeholder
        
    def __str__(self):
        return f"FimObj(prefix={self.prefix}, suffix={self.suffix}, token={self.token}, placeholder={self.placeholder})"
    
    def to_list(self):
        return [self.prefix, self.suffix, self.token, self.placeholder]
    
fim_placeholder = "<FILL>"      
STARCODER_FIM = FimObj("<fim_prefix>", "<fim_suffix>","<fim_middle>", fim_placeholder)


def placeholder_to_std_fmt(prompt : str, fim : FimObj) -> str:
    """
    Take a prompt in fill format and convert it to standard format
    
    >>> "def func(n : <FILL>)"
    "<prefix>def func(n : <suffix>)<fim_token>"
    """
    parts = prompt.split(fim.placeholder)
    if len(parts) != 2:
        raise ValueError(f"Prompt does not contain a single placeholder: {parts}")
    prompt = fim.prefix + parts[0] + fim.suffix + parts[1] + fim.token
    return prompt

def std_to_placeholder_fmt(prompt : str, fim : FimObj) -> str:
    """
    Take a prompt in standard format and convert it to fill format
    
    >>> "<prefix>def func(n : <suffix>)<fim_token>"
    "def func(n : <FILL>)"
    
    """
    return prompt.replace(fim.prefix, "").replace(fim.suffix, fim.placeholder).replace(fim.token,"")

def unfim(text : str, fim : FimObj) -> str:
    """
    Remove fim special tokens and unscramble the prompt
    """
    prefix = text.split(fim.prefix)[-1].split(fim.suffix)[0]
    suffix = text.split(fim.suffix)[-1].split(fim.token)[0]
    middle = text.split(fim.token)[-1]
    return prefix+middle+suffix

def get_captures(prompt : str | tree_sitter.Tree, 
                 query: str, 
                 ignore_parents : List[str] = [],
                 language : str = "typescript") -> List[tree_sitter.Node]:
    """
    Get captures for a prompt given a query
    Ignores any captures whose parents are in ignore_parents
    """
    parser = lang_to_parser[language]
    lang = lang_to_builder[language]
    if isinstance(prompt, str):
        tree = parser.parse(bytes( prompt, "utf8"))
    else:
        tree = prompt
    query = lang.query(query)
    captures = query.captures(tree.root_node)
    
    def matches_any(s : str, patterns : List[str]) -> bool:
        for p in patterns:
            if re.match(p, s):
                return True
        return False
    
    if len(ignore_parents) > 0:
        captures = [c for c in captures if not matches_any(c[0].parent.type,ignore_parents)]
    return captures

def get_builtins_regex(language : str) -> str:
    """
    Returns the builtins for a language as a regex pattern
    """
    if language in ["python", "py"]:
        parent_dir = Path(__file__).parent
        with open(f"{parent_dir}/py_builtins.json","r") as f:
            builtins = json.load(f)
        return "^(" + "|".join(builtins) + ")$"
    elif language in ["typescript", "ts"]:
        raise NotImplementedError("Typescript builtins not implemented")
    

def remove_comments(program : str, 
                    comment_query : str = """((comment) @comment)""",
                    language : str = "typescript") -> str:
    lang = lang_to_builder[language]
    parser = lang_to_parser[language]
    comment_query = lang.query(comment_query)
    tree = parser.parse(bytes(program, "utf8"))
    captures = comment_query.captures(tree.root_node)
    # sort by start byte descending
    captures.sort(key=lambda x: x[0].start_byte, reverse=True)
    program = tree.text
    for c in captures:
        program = replace_between_bytes(program, c[0].start_byte, c[0].end_byte, "")
    return program.decode("utf-8").strip()

def replace_between_bytes(byte_string : bytes,
                           start_byte : int, 
                           end_byte : int,
                           replacement : str = "") -> bytes:
    '''
    Replace tree-sitter interval (start_point, end_point) from a string.
    Inclusive of start_point and end_point
    '''
    byte_replacement = replacement.encode("utf-8")
    if replacement == "":
        modified_byte_string = (
            byte_string[:start_byte] + byte_string[end_byte:]
        )
    else:
        modified_byte_string = (
            byte_string[:start_byte] + byte_replacement + byte_string[end_byte:]
        )
    return modified_byte_string

def find_between_bytes(
    byte_string : bytes,
    start_byte : int, 
    end_byte : int,
    target : str) -> int:
    '''
    Find the first occurence of target between start_byte and end_byte
    '''
    target = bytes(target, "utf-8")
    for i in range(start_byte, end_byte):
        if byte_string[i:i+len(target)] == target:
            return i
    return -1

