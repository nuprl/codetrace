"""
Module for renaming a set of variables in a tyepscript program
using tree-sitter

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
import sys
from multiprocessing import cpu_count
from argparse import ArgumentParser

TS_IDENTIFIER_QUERY = """((identifier) @name)""" 
PY_IDENTIFIER_QUERY = f"""((identifier) @name (#not-match? @name "{get_builtins_regex('python')}"))"""

lang_to_id_query = {
    "typescript" : TS_IDENTIFIER_QUERY,
    "python" : PY_IDENTIFIER_QUERY,
    "ts" : TS_IDENTIFIER_QUERY,
    "py" : PY_IDENTIFIER_QUERY
}

def _get_language(ds: datasets.Dataset) -> str:
    """
    Get the language of the dataset
    """
    if "lang" in ds.column_names:
        assert len(set(ds["lang"])) == 1
        return ds[0]["lang"].lower()
    else:
        assert len(set(ds["language"])) == 1
        return ds[0]["language"].lower()
    
def capture_varnames(tree, language : str = "typescript") -> dict[str, list[tree_sitter.Node]]:
    """
    Given a program, capture all the variable names and their locations
    as tree-sitter (start, end) points
    """
    idquery = lang_to_id_query[language]
    
    ignore_parents = []
    if language in ["python", "py"]:
        ignore_parents = ["dotted_name",r".*type.*"]
        
    captures = get_captures(tree, idquery, ignore_parents, language)
    vars_to_locs = defaultdict(list)
    for c in captures:
        name = c[0].text
        vars_to_locs[name].append(c[0])
    return {k.decode("utf-8"): v for k,v in vars_to_locs.items()}


def rename_variable(program : bytes,
                new_name : str,
               var_locations : list[tree_sitter.Node]) -> bytes:
    """
    Rename a variable in program in bottom-up order to maintain location integrity
    NOTE: only works when new_name is same length as varname
    """
    # sort locations by start byte descending
    var_locations.sort(key=lambda x: x.start_byte, reverse=True)
    
    # replace each varname with a new name
    for capture in var_locations:
        program = replace_between_bytes(program, capture.start_byte, capture.end_byte, new_name)
        
    return program

def make_new_name(new_length : int, existing_names : set[str]) -> str | None:
    """
    Given a set of var captures and a variable name, make a new name for the variable that
    is not already in the program.
    Scrambles the varname until it is not in the program.
    TODO: other strategies for renaming
    - `uniq_` prefix
    - permute the order of the characters
    """
    letters = "abcdefghijklmnopqrstuvwxyz"*10
    new_name = "".join(random.sample(letters, new_length))
    tries = 0
    while new_name in existing_names:
        tries += 1
        if tries > 10000:
            return None
        new_name = "".join(random.sample(letters, new_length))
    return new_name
        
def _predict(llm: LLM, prompt: str | List[str]) -> List[str]:
    """
    Helper function to predict completions
    """
    params = SamplingParams(temperature=0)
    out = llm.generate(prompt, params, use_tqdm=False)
    return [i.outputs[0].text.strip() for i in out]


def dataset_rename_vars(dataset: datasets.Dataset) -> datasets.Dataset:
    """
    For each example in the dataset, rename all variables incrementally
    """
    language = _get_language(dataset)
    parser = lang_to_parser[language]
    lang = lang_to_builder[language]
    
    new_dataset = []
    
    for i,ex in enumerate(tqdm(dataset)):
        fim_program = ex["fim_program"]
        solution = ex["fim_type"]
        
        tree = parser.parse(bytes( fim_program, "utf8"))
        var_locs = capture_varnames(tree, language=language)
        
        names, newnames = set(var_locs.keys()), set()
        
        fim_program = tree.text
        
        for varname, locs in var_locs.items():
            new_name = make_new_name(len(locs[0].text), names)
            if new_name is None or varname[0].isupper() or varname == solution:
                # TODO: there's some tree-sitter bug where it's capturing types as vars
                continue
            
            fim_program_new = rename_variable(fim_program, new_name, locs)
            try:
                fim_program_new.decode("utf-8")
            except:
                continue
            fim_program = fim_program_new
            
            names.add(new_name)
            newnames.add((varname, new_name))
            
            # save old ex
            new_dataset.append({**ex,
                "renamed_fim_program" : fim_program.decode("utf-8"),
                "renamed_variables" : list(newnames),
                "renamed_percent" : len(newnames) / len(var_locs),
            })
            
        
    new_dataset = datasets.Dataset.from_pandas(pd.DataFrame(new_dataset))
    # drop columns "correct" and "overfull"
    new_dataset = new_dataset.remove_columns(["correct", "overfull"])
    return new_dataset

def _filter_incorrect(ds: datasets.Dataset, llm: LLM) -> datasets.Dataset:
    """
    Filter out examples where the model's prediction is incorrect
    """
    def is_incorrect(prediction, solution):
        # make sure prediction did not go overfull and is not the same as the solution
        return prediction != solution and not prediction.startswith(solution)
    
    predictions = []
    params = SamplingParams(temperature=0)
    prompts = ds.map(lambda x: {"_":placeholder_to_std_fmt(x["renamed_fim_program"], STARCODER_FIM)}, num_proc=cpu_count())
    prompts = prompts["_"]

    generations = llm.generate(prompts, params)
    new_ds = []

    for i,output in enumerate(generations):
        generated_text = output.outputs[0].text.strip()
        if is_incorrect(generated_text, ds[i]["fim_type"]):
            new_ds.append({**ds[i], "renamed_generated_text": generated_text})
            
    new_ds = datasets.Dataset.from_pandas(pd.DataFrame(new_ds))
    return new_ds


def _preprocess(dataset : datasets.Dataset, remove_comments=False) -> datasets.Dataset:
    """
    Preprocess the dataset
    """
    language = _get_language(dataset)
        
    dataset = dataset.filter(lambda x: x["correct"] == True and x["overfull"] == False)
    parser = lang_to_parser[language]
    lang = lang_to_builder[language]
    
    # remove examples with:
    # shorthand_property_identifier, shorthand_property_identifier_pattern
    
    preproc_query = """
    ((shorthand_property_identifier_pattern) @sp)
    ((shorthand_property_identifier) @si)
    """
    preproc_query = lang.query(preproc_query)
    
    def _has_captures(prog: str) -> bool:
        tree = parser.parse(bytes(prog, "utf8"))
        captures = preproc_query.captures(tree.root_node)
        return len(captures) > 0
    
    dataset = dataset.filter(lambda x: not _has_captures(x["fim_program"]))
    
    # remove comments
    if remove_comments:
        dataset = dataset.map(lambda x: {"fim_program": remove_comments(x["fim_program"])})
    
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
        return not_type_declaration(x) and not_array_equivalent(x)
    dataset = dataset.filter(_filter)
    return dataset

    
def main(args):
    ds = datasets.load_dataset(args.completions_ds, split=args.split)
    ds = _preprocess(ds, remove_comments=args.remove_comments)
    llm = LLM(args.model)
    ds = dataset_rename_vars(ds)
    ds.push_to_hub(args.new_ds_name + "_unfiltered")
    # sample 3000
    # ds = datasets.Dataset.from_pandas(ds.to_pandas().sample(3000, random_state=42))
    # filter renamed % above threshold
    # ds = ds.filter(lambda x: x["renamed_percent"] >= 0.5)
    print(ds)
    
    ds = _filter_incorrect(ds, llm)
    ds = _postprocess(ds)
    print(ds, len(list(set(ds["hexsha"]))))
    ds.push_to_hub(args.new_ds_name)
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--completions-ds", type=str, required=True)
    parser.add_argument("--model", type=str, default="/home/arjun/models/starcoderbase-1b")
    parser.add_argument("--new-ds-name", type=str, required=True)
    parser.add_argument("--remove-comments", action="store_true")
    parser.add_argument("--lang", type=str, default="typescript")
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()
    main(args)