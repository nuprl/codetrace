from codetrace.type_inf_exp.rename_vars import *
from codetrace.utils import *
import datasets
import pandas as pd
import argparse
from multiprocessing import Pool, cpu_count
from transformers import AutoTokenizer
import torch

def _filter_incorrect(ds: datasets.Dataset, llm: LLM, tokenizer) -> datasets.Dataset:
    """
    Filter out examples where the model's prediction is incorrect. Truncate generation and
    solution at 1 token
    """
    params = SamplingParams(temperature=0, max_tokens=1)
    new_ds = []
    batch_size = 1000
    
    ds = ds.map(lambda x: {"solution":tokenizer.convert_tokens_to_string(tokenizer.tokenize(x["fim_type"])[:1]),
                        "prompt": placeholder_to_std_fmt(x["fim_program"], STARCODER_FIM)}, 
                        num_proc=cpu_count(), keep_in_memory=True)
    print(ds)
    prompts = ds["prompt"]
    
    # batch generations because of RAM
    for i in tqdm(range(0, len(ds), batch_size), desc="Batch generations"):
        generations = llm.generate(prompts[i:i+batch_size], params, use_tqdm=False)

        for j,output in enumerate(generations):
            generated_text = output.outputs[0].text.strip()
            if generated_text != ds[i+j]["solution"]:
                new_ds.append({**ds[i+j],"renamed_generated_text": generated_text})
                
        if len(new_ds) >0 and len(new_ds) % 100 == 0:
            print(f"Saving {len(new_ds)} completions")
            new_ds = datasets.Dataset.from_pandas(pd.DataFrame(new_ds))
            new_ds.save_to_disk(f"temp_{i}")
    
    new_ds = datasets.Dataset.from_pandas(pd.DataFrame(new_ds))
    new_ds = new_ds.remove_columns(["prompt", "solution"])
    return new_ds


def _preprocess(dataset : datasets.Dataset, 
                language: str,
                remove_comments=False) -> datasets.Dataset:
    """
    Preprocess the dataset
    """
    dataset = dataset.filter(lambda x: x["correct"] == True)
    parser = lang_to_parser[language]
    lang = lang_to_builder[language]
    
    if language in ["typescript", "ts"]:
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
    
def _postprocess(dataset : datasets.Dataset, language : str) -> datasets.Dataset:
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
    
    def _ts_filter(x):
        return not_type_declaration(x) and not_array_equivalent(x)
    
    if language in ["typescript", "ts"]:
        dataset = dataset.filter(_ts_filter)
    return dataset

    
def main(args):
    ds = datasets.load_dataset(args.completions_ds, split=args.split)
    ds = _preprocess(ds, language=args.lang, remove_comments=args.remove_comments)
    ds = dataset_rename_vars(ds, language=args.lang)
    ds.push_to_hub(args.new_ds_name + "_unfiltered")
    print(ds)
    
    # ds = datasets.load_dataset(args.new_ds_name + "_unfiltered", split=args.split)
    llm = LLM(args.model, tensor_parallel_size=torch.cuda.device_count())
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    ds = _filter_incorrect(ds, llm, tokenizer)
    ds = _postprocess(ds, language=args.lang)
    ds.push_to_hub(args.new_ds_name)
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--completions-ds", type=str, required=True)
    parser.add_argument("--model", type=str, default="/home/arjun/models/starcoderbase-1b")
    parser.add_argument("--new-ds-name", type=str, required=True)
    parser.add_argument("--remove-comments", action="store_true")
    parser.add_argument("--lang", type=str)
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()
    main(args)