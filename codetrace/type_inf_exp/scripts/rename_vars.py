"""
Module for renaming a set of variables / types in a tyepscript program
using tree-sitter

Need to:
0. start with stenotype FIM prompts for starcoder-1b, get correct ones
1. capture all varnames + their locations in the program
2. rename all varnames to a new name
3. run completions on the new dataset

TODO: make sure not capturing enums properly

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

IDENTIFIER_QUERY = """((identifier) @name)""" # do not capture enums, classes
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

def make_new_name(varname : str, var_captures : dict[str, list[tree_sitter.Node]]) -> str | None:
    """
    Given a set of var captures and a variable name, make a new name for the variable that
    is not already in the program.
    Scrambles the varname until it is not in the program.
    """
    # if is not bytes literal, convert to bytes literal
    existing_names = set(var_captures.keys())
    # scramble varname
    random.seed(42)
    new_name = varname
    tries = 0
    while new_name in existing_names:
        tries += 1
        if tries > 100:
            return None
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
    For each example in the dataset, rename all variables until the llm's type prediction
    for the fim token breaks
    """
    new_dataset = []
    for i,ex in enumerate(tqdm(dataset)):
        program = ex["fim_prog"]
        solution = ex["solution"]
        prediction = solution
        var_locs = capture_varnames(program)
        for varname, locs in var_locs.items():
            varname = varname.decode("utf-8")
            new_name = make_new_name(varname, var_locs)
            if new_name is None:
                continue
            # if varname starts with capital letter, it is an enum identifier
            if varname[0].isupper():
                # hacky way of checking that enum identifiers are not renamed
                continue
            program = rename_variable(program, new_name, locs)
            # run the llm on the new program
            prediction = _predict(llm, placeholder_to_std_fmt(program, STARCODER_FIM))[0]
            prediction = prediction.strip()
            if prediction != solution and not prediction.startswith(solution):
                ex = {"new_generated": prediction, **ex, "renamed_prog": program}
                new_dataset.append(ex)
                break
        
    new_dataset = datasets.Dataset.from_pandas(pd.DataFrame(new_dataset))
    return new_dataset

def _preprocess(dataset : datasets.Dataset) -> datasets.Dataset:
    """
    Preprocess the dataset
    """
    dataset = dataset.filter(lambda x: x["correctness"] == "correct")
    dataset = dataset.map(lambda x: {"fim_prog": std_to_placeholder_fmt(x["prompt"], STARCODER_FIM)})
    
    # rename / remove:
    # shorthand_property_identifier

    # remove:
    # shorthand_property_identifier_pattern

    # ignore:
    # nested_identifier
    # statement_identifier
    
    preproc_query = """
    ((shorthand_property_identifier_pattern) @sp)
    ((shorthand_property_identifier) @si)
    """
    preproc_query = TS_LANGUAGE.query(preproc_query)
    
    def _has_captures(prog: str) -> bool:
        tree = TS_PARSER.parse(bytes(prog, "utf8"))
        captures = preproc_query.captures(tree.root_node)
        return len(captures) > 0
    
    # TODO remove comments
    
    dataset = dataset.filter(lambda x: not _has_captures(x["fim_prog"]))
    return dataset
    
def _postprocess(dataset : datasets.Dataset) -> datasets.Dataset:
    """
    Postprocess the dataset. Make sure new_generated is not the same as solution
    inner type
    """
    def condition(x):
        type_declaration = x["solution"] + "="+ x["new_generated"]
        # if the order of non-whitespace and non-alphanumerical characters is the same, then the strings are the same
        matches = re.findall(r"\S", type_declaration)
        type_declaration = "".join(matches)
        matches_in_prog = re.findall(r"\S", x["original_prog"])
        matches_in_prog = "".join(matches_in_prog)
        int_type_declaration = type_declaration.replace("=", "").replace("}", ";}")
        return not type_declaration in matches_in_prog and not int_type_declaration in matches_in_prog

    # todo remove number[] and Array<number>
    def condition2(x):
        if "[]" in x["new_generated"] and "Array<" in x["solution"]:
            # capture alphanum chars
            matches = re.findall(r"\w", x["new_generated"])
            new_generated = "".join(matches)
            matches = re.findall(r"\w", x["solution"])
            solution = "".join(matches).replace("Array", "")
            return new_generated != solution
        return True
    
    dataset = dataset.filter(lambda x: condition(x) and condition2(x))
    return dataset

def _join_with_original(dataset: datasets.Dataset, original: datasets.Dataset) -> datasets.Dataset:
    """
    Join the renamed dataset with the original dataset program
    """
    new_ds = []
    hexsha_to_prog = {x["hexsha"]: x["content"] for x in original}
    for i,ex in enumerate(tqdm(dataset)):
        original_ex = hexsha_to_prog[ex["hexsha"]]
        ex = {**ex, "original_prog": original_ex}
        new_ds.append(ex)
    return datasets.Dataset.from_pandas(pd.DataFrame(new_ds))

def _reformat(dataset : datasets.Dataset) -> datasets.Dataset:
    """
    Reformat the dataset to be in the column format:
    generated,solution,hexsha,prompt,correctness,id,original_prog
    """
    new_ds = []
    for i,ex in enumerate(tqdm(dataset)):
        incorrect = {"generated": ex["new_generated"], 
                     "solution": ex["solution"], 
                     "hexsha": ex["hexsha"], 
                     "prompt": placeholder_to_std_fmt(ex["renamed_prog"], STARCODER_FIM), 
                     "correctness": "incorrect",
                     "id": ex["id"],
                     "original_prog": ex["original_prog"],
                     "renamed" : True}
        correct = {"generated": ex["generated"], 
                   "solution": ex["solution"], 
                   "hexsha": ex["hexsha"], 
                   "prompt": ex["prompt"], 
                   "correctness": ex["correctness"],
                   "id": ex["id"],
                   "original_prog": ex["original_prog"],
                   "renamed" : False}
        new_ds.append(incorrect)
        new_ds.append(correct)
    return datasets.Dataset.from_pandas(pd.DataFrame(new_ds))

if __name__ == "__main__":
    # llm = LLM("/home/arjun/models/starcoderbase-1b")
    # dataset = datasets.load_dataset("franlucc/starcoderbase-1b-completions_typeinf_analysis_v1", split="train")
    
    # # dataset = dataset.select(range(300))
    # dataset = _preprocess(dataset)
    # new_dataset = rename_vars_until_break(dataset, llm)
    # # new_dataset = datasets.load_dataset("franlucc/stenotype-eval-renamed-v1", split="train")
    # new_dataset = _postprocess(new_dataset)
    # new_dataset.push_to_hub("franlucc/stenotype-eval-renamed-v3")
    
    new_dataset = datasets.load_dataset("franlucc/stenotype-eval-renamed-v3", split="train")
    new_dataset = _reformat(new_dataset)
    new_dataset.push_to_hub("franlucc/stenotype-eval-renamed-v4")
    