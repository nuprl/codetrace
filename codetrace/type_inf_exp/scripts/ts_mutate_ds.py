import datasets
import argparse
from codetrace.type_inf_exp import ts_mutator
from codetrace.utils import *
import pandas as pd
from multiprocessing import Pool, cpu_count
from transformers import AutoTokenizer
import torch
import json
from vllm import LLM, SamplingParams
from multiprocessing import cpu_count
from tqdm import tqdm
from codetrace.fast_utils import get_batches_fast, batched_do_func
import os

def filter_incorrect(ds: datasets.Dataset, 
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
    ds = ds.map(lambda x: {"prompt" : placeholder_to_std_fmt(x["mutated_program"], STARCODER_FIM),
                            "solution":tokenizer.decode(tokenizer.encode(x["fim_type"])[0])}, desc="Prepping prompts")
    prompts = ds["prompt"]
    # batch generations so we can save them early
    for n,i in tqdm(enumerate(range(0, len(ds), batch_size)), desc="Batch generations", total=len(ds) // batch_size):
        
        generations = llm.generate(prompts[i:i+batch_size], params, use_tqdm=False)

        for j,output in enumerate(generations):
            generated_text = output.outputs[0].text.strip()
            if generated_text != ds[i+j]["solution"]:
                new_ds.append({**ds[i+j],"mutated_generated_text": generated_text})
                
        # save every
        print(f"Len new_ds: {len(new_ds)}")
        new_ds_hf = datasets.Dataset.from_pandas(pd.DataFrame(new_ds))
        new_ds_hf.push_to_hub(new_ds_name, private=True)
    
    new_ds = datasets.Dataset.from_pandas(pd.DataFrame(new_ds))
    new_ds.push_to_hub(new_ds_name, private=True)
    new_ds = new_ds.remove_columns(["prompt", "solution"])
    return new_ds


def ts_preprocess(iterable, correct_bool = True):
    """
    Preprocess the dataset
    - Take only correct examples
    - TODO: currently do not support shorthands, so either unroll or remove 
        shorthand_property_identifier, shorthand_property_identifier_pattern
    """
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
    
    def _condition(x):
        return x["correct"] == correct_bool and not _has_captures(x["fim_program"])
    
    if isinstance(iterable, datasets.Dataset):
        return iterable.filter(_condition, desc="Preprocess")
    else:
        return [i for i in iterable if _condition(i)]

def preprocess_then_mutate(batch, mutations, correct_bool = True):
    post = ts_preprocess(batch, correct_bool)
    return ts_mutator.iter_apply_random_mutations(post, mutations)
    
def main(args):
    if "do_mutate" in args.actions:
        ds = datasets.load_dataset(args.completions_ds, split=args.split)
        if args.max_size > -1:
            ds = ds.shuffle(seed=42).select(range(args.max_size))
        mutations = [getattr(ts_mutator, m) for m in args.mutations]
        
        batches = get_batches_fast(ds, len(ds), cpu_count())
        results = batched_do_func(batches, cpu_count(), preprocess_then_mutate, mutations=mutations, correct_bool=args.correct_bool)

        def _yielder():
            for ex in tqdm(results, desc="Yielding", total=len(results)):
                yield ex
                
        ds = datasets.Dataset.from_generator(_yielder)
        print(ds)
        ds.push_to_hub(args.new_ds_name + "_" + args.model_name + "_unfiltered", private=True)
    else:
        ds = datasets.load_dataset(args.new_ds_name + "_" + args.model_name + "_unfiltered", split=args.split)
        
    if "do_completions" in args.actions:
        llm = LLM(args.model)
        ds = filter_incorrect(ds, llm, args.new_ds_name + "_" + args.model_name)
        print(ds)
        ds.push_to_hub(args.new_ds_name + "_" + args.model_name, private=True)
    
if __name__ == "__main__":
    datasets.disable_caching()
    print("Caching:", datasets.is_caching_enabled())
    parser = argparse.ArgumentParser()
    parser.add_argument("--completions-ds", type=str, required=True)
    parser.add_argument("--model", type=str, default="/home/arjun/models/starcoderbase-1b")
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--new-ds-name", type=str, required=True)
    parser.add_argument("--mutations", type=str, required=True, nargs="+", choices=["mutation_rename_type",
                                                                                    "mutation_rename_vars",
                                                                                    "mutation_delete_annotation"])
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max-size", type=int, default=-1)
    parser.add_argument("--correct-bool", type=bool, default=True)
    parser.add_argument("--actions", nargs="+", choices=["do_completions", "do_mutate"], default=["do_completions", "do_mutate"])
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    print(os.environ["CUDA_VISIBLE_DEVICES"])
    main(args)

    