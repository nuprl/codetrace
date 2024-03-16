from codetrace.type_inf_exp.rename_vars import *
from codetrace.utils import *
import datasets
import pandas as pd
import argparse
from multiprocessing import Pool, cpu_count
from transformers import AutoTokenizer
import torch
import json
import hashlib

def _filter_incorrect(ds: datasets.Dataset, 
                      llm: LLM,
                      new_ds_name,
                      batch_size = 10000) -> datasets.Dataset:
    """
    Filter out examples where the model's prediction is incorrect. Truncate generation and
    solution at 1 token
    """
    tokenizer = llm.get_tokenizer().tokenizer
    params = SamplingParams(temperature=0, max_tokens=1)
    new_ds = []
    ds = ds.map(lambda x: {"prompt" : placeholder_to_std_fmt(x["renamed_fim_program"], STARCODER_FIM),
                            "solution":tokenizer.decode(tokenizer.encode(x["fim_type"])[0])})
    prompts = ds["prompt"]
    # batch generations because of RAM
    for i in tqdm(range(0, len(ds), batch_size), desc="Batch generations"):
        
        generations = llm.generate(prompts[i:i+batch_size], params)

        for j,output in enumerate(generations):
            generated_text = output.outputs[0].text.strip()
            if generated_text != ds[i+j]["solution"]:
                new_ds.append({**ds[i+j],"renamed_generated_text": generated_text})
                
        if i % 1000 == 0:
            # save every 1000 examples
            print(f"Len new_ds: {len(new_ds)}")
            new_ds_hf = datasets.Dataset.from_pandas(pd.DataFrame(new_ds))
            new_ds_hf.push_to_hub(new_ds_name)
    
    new_ds = datasets.Dataset.from_pandas(pd.DataFrame(new_ds))
    new_ds.push_to_hub(new_ds_name)
    new_ds = new_ds.remove_columns(["prompt", "solution"])
    return new_ds


def _ts_preprocess(dataset : datasets.Dataset) -> datasets.Dataset:
    """
    Preprocess the dataset
    - Take only correct examples
    - TODO: currently do not support shorthands, so either unroll or remove 
        shorthand_property_identifier, shorthand_property_identifier_pattern
    """
    dataset = dataset.filter(lambda x: x["correct"] == True)
    parser = lang_to_parser["ts"]
    lang = lang_to_builder["ts"]
    
    # remove examples with:
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
    
    return dataset
    
def _ts_postprocess(dataset : datasets.Dataset) -> datasets.Dataset:
    """
    # TODO: this is hacky
    Postprocess the dataset. Make sure new_generated is not the same as the solution
    inner type. This is because this is not ``really`` incorrect
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
        "Array<number>", then this is not an example we wish to keep.
        Vice versa is also true. i.e. if model generates "Array<number>" and correct
        solution is "number[]".
        """
        if "[]" in x["generated_text"] and "Array<" in x["fim_type"]:
            # capture alphanum chars
            matches = re.findall(r"\w", x["generated_text"])
            new_generated = "".join(matches)
            matches = re.findall(r"\w", x["fim_type"])
            solution = "".join(matches).replace("Array", "")
            return new_generated != solution
        elif "Array<" in x["generated_text"] and "[]" in x["fim_type"]:
            matches = re.findall(r"\w", x["generated_text"])
            new_generated = "".join(matches).replace("Array", "")
            matches = re.findall(r"\w", x["fim_type"])
            solution = "".join(matches)
            return new_generated != solution
        return True
    
    def _ts_filter(x):
        return not_type_declaration(x) and not_array_equivalent(x)
    
    dataset = dataset.filter(_ts_filter)
    return dataset

    
def main(args):
    ds = datasets.load_dataset(args.completions_ds, split=args.split)
    ds = _ts_preprocess(ds)
    ds = dataset_incremental_rename_vars(ds)
    ds.push_to_hub(args.new_ds_name + "_unfiltered")
    print(ds)
    
    # ds = datasets.load_dataset(args.new_ds_name + "_unfiltered", split=args.split)
    llm = LLM(args.model)
    ds = _filter_incorrect(ds, llm, args.new_ds_name)
    ds = _ts_postprocess(ds)
    if "hexsha" not in ds.column_names:
        ds = ds.add_column("hexsha", [hashlib.sha256(bytes(x["zip"]+x["filename"],"utf-8")).hexdigest() for x in ds])
    ds.push_to_hub(args.new_ds_name)
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--completions-ds", type=str, required=True)
    parser.add_argument("--model", type=str, default="/home/arjun/models/starcoderbase-1b")
    parser.add_argument("--new-ds-name", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()
    main(args)