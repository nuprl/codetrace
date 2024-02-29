from codetrace.type_inf_exp.scripts.rename_vars import _get_language, lang_to_parser, lang_to_builder, capture_varnames, rename_variable, make_new_name
import datasets
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser

def dataset_rename_vars(dataset: datasets.Dataset) -> datasets.Dataset:
    """
    For each example in the dataset, rename all variables incrementally
    """
    language = _get_language(dataset)
    parser = lang_to_parser[language]
    lang = lang_to_builder[language]
    
    new_dataset = []
    
    for i,ex in enumerate(tqdm(dataset)):
        program = ex["prompt"]
        
        tree = parser.parse(bytes( program, "utf8"))
        var_locs = capture_varnames(tree)
        
        names, newnames = set(var_locs.keys()), set()
        
        program = tree.text
        
        for varname, locs in var_locs.items():
            new_name = make_new_name(len(locs[0].text), names)
            
            program_new = rename_variable(program, new_name, locs)
            try:
                program_new.decode("utf-8")
            except:
                continue
            program = program_new
            
            names.add(new_name)
            newnames.add((varname, new_name))
            
            # save old ex
            new_dataset.append({**ex,
                "renamed_program" : program.decode("utf-8"),
                "renamed_variables" : list(newnames),
                "renamed_percent" : len(newnames) / len(var_locs),
            })
            
        
    new_dataset = datasets.Dataset.from_pandas(pd.DataFrame(new_dataset))
    return new_dataset

def _preprocess(ds: datasets.Dataset) -> datasets.Dataset:
    """
    Preprocess the dataset, return only correct programs
    """
    return ds.filter(lambda x: x["correct"])

def main(args):
    ds = datasets.load_dataset(args.completions_ds, split=args.split)
    ds = _preprocess(ds)
    ds = dataset_rename_vars(ds)
    print(ds)
    ds.push_to_hub(args.new_ds_name)
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--completions-ds", type=str, required=True)
    parser.add_argument("--model", type=str, default="/home/arjun/models/starcoderbase-1b")
    parser.add_argument("--new-ds-name", type=str, required=True)
    parser.add_argument("--lang", type=str, default="python")
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()
    main(args)