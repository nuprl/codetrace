import datasets
import argparse
from codetrace.utils import STARCODER_FIM, placeholder_to_std_fmt, CODELLAMA_FIM
import pandas as pd
from multiprocessing import Pool, cpu_count
from transformers import AutoTokenizer
import torch
import json
from vllm import LLM, SamplingParams
from multiprocessing import cpu_count
from tqdm import tqdm
from codetrace.fast_utils import get_batches_fast, batched_do_func
from codetrace.py_mutator import random_mutations_subset, NONVAR_STATEMENTS

def filter_incorrect(ds: datasets.Dataset, 
                      llm: LLM,
                      new_ds_name,
                      batch_size = 60000) -> datasets.Dataset:
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
        new_ds_hf.push_to_hub(new_ds_name)
    
    new_ds = datasets.Dataset.from_pandas(pd.DataFrame(new_ds))
    new_ds.push_to_hub(new_ds_name)
    new_ds = new_ds.remove_columns(["prompt", "solution"])
    return new_ds

def add_type_aliases_after_imports(code: bytes, type_aliases : bytes) -> bytes:
    """
    Add type aliases to the prefix after the last import statement
    NOTE:we assume all imports are at the top of the file
    """
    import_statement_query = " ".join([f"(({n}) @imp)" for n in NONVAR_STATEMENTS if "import" in n])
    captures = get_captures(code, import_statement_query, [], "py")
    if len(captures) == 0:
        return type_aliases + code
    # find the last import statement
    last_import = max(captures, key=lambda x: x[0].end_byte)
    new_code = code[:last_import[0].end_byte] + b"\n" + type_aliases + code[last_import[0].end_byte:]
    return new_code

def batched_random_mutate_subset(batch, mutations):
    new_batch = []
    seen = set() # dedup
    for ex in batch:
        prefix = ex["prefix"]
        suffix = ex["suffix"]
        middle = ex["middle"]
        loc = len(bytes(prefix, "utf-8"))
        for (new_target_index, type_aliases, candidate_code) in random_mutations_subset(prefix+middle+suffix, loc, target=middle, target_mutations=mutations):
            suffix_start = new_target_index + len(bytes(middle, "utf-8"))
            prefix_new = candidate_code[:new_target_index]
            suffix_new = candidate_code[suffix_start:]
            prefix_new = add_type_aliases_after_imports(prefix_new, type_aliases)
            mutated_program = (prefix_new + b"<FILL>" + suffix_new).decode("utf-8")
            if mutated_program not in seen:
                new_batch.append({
                    "mutated_program": mutated_program,
                    "mutations": mutations,
                    **ex
                })
            seen.add(mutated_program)
    return new_batch

def py_preprocess(iterable):
    """
    Preprocess the dataset
    - Take only correct examples
    """
    def _condition(x):
        return x["correct"] == True
    
    if isinstance(iterable, datasets.Dataset):
        return iterable.filter(_condition, desc="Preprocess")
    else:
        return [i for i in iterable if _condition(i)]

def preprocess_then_mutate(batch, mutations):
    post = py_preprocess(batch)
    return batched_random_mutate_subset(post, mutations)

def main(args):
    if not args.only_completions:
        ds = datasets.load_dataset(args.completions_ds, split=args.split)
        if args.max_size > -1:
            ds = ds.shuffle(seed=42).select(range(args.max_size))
        
        batches = get_batches_fast(ds, len(ds), cpu_count())
        results = batched_do_func(batches, cpu_count(), preprocess_then_mutate, mutations=args.mutations)

        def _yielder():
            for ex in tqdm(results, desc="Yielding", total=len(results)):
                yield ex
                
        ds = datasets.Dataset.from_generator(_yielder)
        print(ds)
        ds.push_to_hub(args.new_ds_name + "_unfiltered")
    
    if args.only_completions:
        ds = datasets.load_dataset(args.new_ds_name + "_unfiltered", split=args.split)
        if args.max_size > -1:
            ds = ds.shuffle(seed=42).select(range(args.max_size))
    
    llm = LLM(args.model, device_map=f"cuda:{args.gpu}")
    print(args.model, args.model_name)
    ds = filter_incorrect(ds, llm, args.new_ds_name + "_" + args.model_name)
    print(ds)
    
    ds.push_to_hub(args.new_ds_name + "_" + args.model_name)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--completions-ds", type=str, required=True)
    parser.add_argument("--model", type=str, default="/home/arjun/models/starcoderbase-1b")
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--new-ds-name", type=str, required=True)
    parser.add_argument("--mutations", type=str, required=True, nargs="+", choices=["rename_types",
                                                                                    "rename_vars",
                                                                                    "remove_type_annotations"])
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max-size", type=int, default=-1)
    parser.add_argument("--only-completions", action="store_true", default=False)
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()
    
    main(args)
