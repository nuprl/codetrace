"""
Module for renaming a set of variables in a tyepscript program
using tree-sitter

TODO: make sure not capturing enums properly
TODO: enable renaming with "uniq_" by dyanmically updating locations of captures
"""
import tree_sitter
from codetrace.utils import *
import datasets
from collections import defaultdict
from vllm import LLM, SamplingParams
import random
from tqdm import tqdm
import pandas as pd
import re
import sys

IDENTIFIER_QUERY = """((identifier) @name)""" 
query = TS_LANGUAGE.query(IDENTIFIER_QUERY)

def capture_varnames(program : str) -> dict[str, list[tree_sitter.Node]]:
    """
    Given a program, capture all the variable names and their locations
    as tree-sitter (start, end) points
    """
    tree = TS_PARSER.parse(bytes( program, "utf8"))
    captures = query.captures(tree.root_node)
    vars_to_locs = defaultdict(list)
    for c in captures:
        name = c[0].text
        vars_to_locs[name].append(c[0])
    return {k.decode("utf-8"): v for k,v in vars_to_locs.items()}


def rename_variable(program : str,
                new_name : str,
               var_locations : list[tree_sitter.Node]) -> str:
    """
    Rename a variable in program in bottom-up order to maintain location integrity
    """
    # sort locations by start byte descending
    var_locations.sort(key=lambda x: x.start_byte, reverse=True)
    
    # replace each varname with a new name
    for capture in var_locations:
        program = replace_between_points(program, capture.start_point, capture.end_point, new_name)
        
    return program

def make_new_name(varname : str, existing_names : set[str]) -> str | None:
    """
    Given a set of var captures and a variable name, make a new name for the variable that
    is not already in the program.
    Scrambles the varname until it is not in the program.
    TODO: other strategies for renaming
    - `uniq_` prefix
    - permute the order of the characters
    """
    random.seed(42)
    new_name = varname
    tries = 0
    while new_name in existing_names:
        tries += 1
        if tries > 100:
            return None
        elif len(new_name) == 1:
            # for variable names of length 1, just pick a random character
            new_name = random.choice("abcdefghijklmnopqrstuvwxyz")
        else:
            new_name = "".join(random.sample(new_name, len(new_name)))
    return new_name
        

def _predict(llm: LLM, prompt: str | List[str]) -> List[str]:
    """
    Helper function to predict completions
    """
    params = SamplingParams(temperature=0)
    out = llm.generate(prompt, params, use_tqdm=False)
    return [i.outputs[0].text for i in out]

def rename_vars_until_break(dataset: datasets.Dataset, 
                            llm: LLM) -> datasets.Dataset:
    """
    For each example in the dataset, rename all variables until the llm's type prediction breaks
    """
    new_dataset = []
    for i,ex in enumerate(tqdm(dataset)):
        fim_program = ex["fim_program"]
        solution = ex["fim_type"]
        var_locs = capture_varnames(fim_program)
        names, newnames = set(var_locs.keys()), set()
        for varname, locs in var_locs.items():
            new_name = make_new_name(varname, names)
            if new_name is None:
                continue
            # if varname starts with capital letter, it is an enum identifier
            # TODO: there's some tree-sitter bug where it's capturing types as vars
            if varname[0].isupper() or varname == solution:
                continue
            
            fim_program = rename_variable(fim_program, new_name, locs)
            names.add(new_name)
            newnames.add(new_name)
            
            # run the llm on the new program
            prediction = _predict(llm, placeholder_to_std_fmt(fim_program, STARCODER_FIM))[0].strip()
            if prediction != solution and not prediction.startswith(solution):
                # save old ex
                new_dataset.append(ex.copy())
                # save new ex
                ex["fim_program"] = fim_program
                ex["generated_text"] = prediction
                ex["correct"] = False
                ex["new_varnames"] = list(newnames)
                new_dataset.append(ex)
                break
        
    new_dataset = datasets.Dataset.from_pandas(pd.DataFrame(new_dataset))
    return new_dataset

def _preprocess(dataset : datasets.Dataset) -> datasets.Dataset:
    """
    Preprocess the dataset
    """
    dataset = dataset.filter(lambda x: x["correct"] == True)
    
    # remove examples with:
    # shorthand_property_identifier, shorthand_property_identifier_pattern
    
    preproc_query = """
    ((shorthand_property_identifier_pattern) @sp)
    ((shorthand_property_identifier) @si)
    """
    preproc_query = TS_LANGUAGE.query(preproc_query)
    
    def _has_captures(prog: str) -> bool:
        tree = TS_PARSER.parse(bytes(prog, "utf8"))
        captures = preproc_query.captures(tree.root_node)
        return len(captures) > 0
    
    # TODO remove comments?
    
    dataset = dataset.filter(lambda x: not _has_captures(x["fim_program"]))
    return dataset
    
def _postprocess(dataset : datasets.Dataset) -> datasets.Dataset:
    """
    # TODO: this is hacky
    Postprocess the dataset. Make sure new_generated is not the same as the solution
    inner type
    """
    def not_type_declaration(x):
        """
        for example if model generates "a | b" and correct solution
        is "TYP" where "TYP = a | b", then this is not an example we wish to keep
        """
        type_declaration = x["fim_type"] + "="+ x["generated_text"]
        # if the order of non-whitespace and non-alphanumerical characters is the same, then the strings are the same
        matches = re.findall(r"\S", type_declaration)
        type_declaration = "".join(matches)
        matches_in_prog = re.findall(r"\S", x["fim_program"])
        matches_in_prog = "".join(matches_in_prog)
        int_type_declaration = type_declaration.replace("=", "").replace("}", ";}")
        return not type_declaration in matches_in_prog and not int_type_declaration in matches_in_prog

    def not_array_equivalent(x):
        """
        for example if model generates "number[]" and correct solution is
        "Array<number>", then this is not an example we wish to keep
        """
        if "[]" in x["generated_text"] and "Array<" in x["fim_type"]:
            # capture alphanum chars
            matches = re.findall(r"\w", x["generated_text"])
            new_generated = "".join(matches)
            matches = re.findall(r"\w", x["fim_type"])
            solution = "".join(matches).replace("Array", "")
            return new_generated != solution
        return True
    
    def _filter(x):
        if x["correct"]:
            return True
        else:
            return not_type_declaration(x) and not_array_equivalent(x)
    dataset = dataset.filter(_filter)
    return dataset

def main():
    newname = sys.argv[1]
    ds = datasets.load_dataset("franlucc/stenotype-type-inference-fim-evaluated", split="train")
    ds = _preprocess(ds)
    llm = LLM("/home/arjun/models/starcoderbase-1b")
    ds = rename_vars_until_break(ds, llm)
    ds = _postprocess(ds)
    ds.push_to_hub(newname)
    
if __name__ == "__main__":
    main()