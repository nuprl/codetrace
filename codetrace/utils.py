import glob
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import os
import json
from tree_sitter import Language, Parser
import tempfile

vendor = "/home/franlucc/llm_libs"
Language.build_library(
    "build/my-languages.so",
    [f"{vendor}/tree-sitter-typescript/typescript"],
)
TS_LANGUAGE = Language("build/my-languages.so", "typescript")
TS_PARSER = Parser()
TS_PARSER.set_language(TS_LANGUAGE)

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

def fim_dataset(hf_dataset):
    """
    Given a huggingface dataset, for each example
    return a list of variations with FIM placeholders
    """
    fim_examples = []
    for ex in tqdm(hf_dataset):
        fim_progs = fim_prog(ex["content"])
        fim_examples.append(fim_progs)
    return fim_examples

def fim_prog(prog : str) -> list[str]:
    """
    Given a typescript program, return a list of variations with FIM placeholders
    """
    tree = TS_PARSER.parse(bytes( prog, "utf8"))
    
    query = TS_LANGUAGE.query(
        """
((type_annotation) @name)
    """
    )
    fim_variations = []
    captures = query.captures(tree.root_node)
    for c in captures:
        s = replace_between_points(prog, c[0].start_point, c[0].end_point, fim_placeholder)
        fim_variations.append(s)
    return fim_variations


def fim_prog_func(prog : str) -> list[Tuple[str]]:
    """
    Given a typescript program, return a list of variations with FIM placeholders.
    Only affect type annotations within function signatures
    """
    tree = TS_PARSER.parse(bytes( prog, "utf8"))
    
    # captures types within functions
    query = TS_LANGUAGE.query(
        """
(required_parameter
      pattern: (_) (type_annotation) @tp)
  (optional_parameter
      pattern: (_) (type_annotation) @tp)
    return_type: (type_annotation) @tp
    """
    )
    fim_variations = []
    captures = query.captures(tree.root_node)
    for c in captures:
        text = c[0].text.decode("utf-8").strip()[1:]
        s = replace_between_points(prog, c[0].start_point, c[0].end_point, fim_placeholder)
        fim_variations.append((s, text))
    return fim_variations

        
def replace_between_points(original_string : str,
                           start_point : Tuple[int], 
                           end_point : Tuple[int],
                           replacement : str = "") -> str:
    '''
    Replace tree-sitter interval (start_point, end_point) from a string.
    Inclusive of start_point and end_point
    '''
    with tempfile.NamedTemporaryFile(mode='w+t', delete=False) as temp:
        temp.write(original_string)
        temp.seek(0)
        original_string = temp.read()

    start_index = point_to_index_loc(start_point, original_string)
    end_index = point_to_index_loc(end_point, original_string)

    modified_string = (
        original_string[:start_index] + replacement + original_string[end_index:]
    )
    return modified_string

def point_to_index_loc(point: Tuple[int], original_string: str) -> int:
    """
    Translate tree-sitter tuple indexing to string int indexing
    """
    row = point[0]
    col = point[1]
    if row == 0:
        return col
    else:
        return len("\n".join(original_string.splitlines()[:row])) + col+1 # for "\n"
    

# from MultiPL-E
def estimator(n: int, c: int, k: int) -> float:
    """
    Calculates 1 - comb(n - c, k) / comb(n, k).
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

def pass_k(gens : List[str], solution : str, k : int):
    """
    Given a list of generated programs for a prompt, and the gold solution for the prompt,
    calculate pass@k
    """
    n = len(gens)
    c = len([i for i in gens if (solution == i)])
    return estimator(n, c, k)