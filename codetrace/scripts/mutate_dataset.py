import asyncio
from argparse import ArgumentParser
from multiprocessing import cpu_count
import os
from pathlib import Path
import shutil
from typing import List,Union,Callable,Dict,Any,Optional,TypeVar,Set
import pandas as pd
from transformers import AutoTokenizer, PreTrainedTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm
import datasets
from datasets import IterableDataset
from codetrace.base_mutator import MutationFn, Mutation
from codetrace.py_mutator import PyMutator
from codetrace.ts_mutator import TsMutator
from codetrace.parsing_utils import get_model_fim,get_captures, FimChat, FimObj, prepare_fim_prompt
from codetrace.utils import (
    load_dataset,
    num_available_devices,
    hex_encode
)
from vllm import AsyncLLMEngine
from codetrace.vllm_utils import (
    load_vllm,
    generate_completions
)

def get_mutations(key: str) -> str:
    if key == "vars":
        return "rename_vars"
    elif key == "types":
        return "rename_types"
    else:
        return "delete_annotations"

def _save(data: List[Dict[str,Any]], path:str, message:str):
    print(message)
    temp_path = Path(str(path) + "_temp")
    new_ds = datasets.Dataset.from_list(data)
    if os.path.exists(path):
        existing_completions = datasets.load_from_disk(path)  
        new_ds = datasets.concatenate_datasets([new_ds, existing_completions])

    # workaround huggingface save_to_disk permissions
    new_ds.save_to_disk(temp_path)
    shutil.rmtree(path, ignore_errors=True)
    shutil.move(temp_path, path)
    shutil.rmtree(temp_path, ignore_errors=True)
    print(f"Collected {len(new_ds)} candidates")

def _mutate_item(program: str, fim_type: str, mutations: List[Mutation], mutate_fn: MutationFn) -> str:
    new_program = None
    for _ in range(10):
        new_program = mutate_fn(program, fim_type, mutations)
        if new_program:
            break
    return new_program

def _preprocess(
    ds: IterableDataset, blacklist: Set[str],
    model_fim: Union[FimObj,FimChat], tokenizer: PreTrainedTokenizer,
    lang:str, mutations: List[str],batch_size:int
) -> IterableDataset:
    """
    Apply the following preprocessing steps asynchronously.
    1. Select correct vanilla completions
    2. For ts, remove programs with shorthands which our mutations do not support
    3. Remove any programs that have already been mutated (in blacklist)
    4. Apply random mutations to create mutated_program field
    """
    if lang == "py":
        _condition = (lambda x: x["correct"])
        mutator = PyMutator()
    else:
        preproc_query = """
        ((shorthand_property_identifier_pattern) @si)
        ((shorthand_property_identifier) @si)
        """
        _condition = (lambda x: x["correct"] and 
                      len(get_captures(x["fim_program"], preproc_query, "ts","si")) == 0)
        mutator = TsMutator()
    
    mutations = [get_mutations(m) for m in mutations]
    ds = ds.filter(lambda x: hex_encode(x["fim_program"]) not in blacklist and _condition(x))
    ds = ds.map(lambda x: {**x, 
            "mutation_names": mutations,
            "mutated_program": _mutate_item(x["fim_program"], x["fim_type"], 
                                        mutations, mutator.random_mutate_ordered_by_type)})
    
    ds = ds.filter(lambda x: x["mutated_program"])
    ds = ds.map(lambda x: {**x, "_prompt": prepare_fim_prompt(tokenizer, model_fim, x["mutated_program"])})

    return ds

def main(
    llm: AsyncLLMEngine,
    tokenizer: PreTrainedTokenizer,
    ds: datasets.IterableDataset,
    output_path: Path,
    model_fim: Union[FimObj,FimChat],
    batch_size: int,
    model_name: str,
    lang:str,
    mutations:List[str],
    max_num_candidates: int
):
    # resume from completions if they exist
    completions, blacklist = [], set()
    if os.path.exists(output_path):
        completions = datasets.load_from_disk(output_path, keep_in_memory=False)
        print(f"Resuming from {len(completions)} completions.")
        print(f"Collected {len(completions)} candidates")
        for row in completions:
            blacklist.add(hex_encode(row["fim_program"]))

    # preprocess data
    ds = _preprocess(ds, blacklist, model_fim, tokenizer, lang, mutations, batch_size)

    # batch generations because of cpu ops in vllm
    num_completed = len(completions)
    for i,batch in tqdm(enumerate(ds.iter(batch_size)), desc="Batch generations"):
        batch_completions = asyncio.run(generate_completions(
                                    llm,
                                    batch,
                                    batch_size,
                                    use_tqdm=(i == 0)
                                ))
        breaking_mutations = []
        for item in batch_completions:
            correct = item["_generated"] == item["fim_type"]
            if not correct:
                breaking_mutations.append({**item, 
                                        "mutated_generated_text": item["_generated"], 
                                        "correct": False,
                                        "model_name": model_name})
        num_completed += len(breaking_mutations)
        # save every batch
        if len(breaking_mutations) > 0:
            _save(breaking_mutations, output_path, f"Saving {i} batch")
        if max_num_candidates > 0 and num_completed >= max_num_candidates:
            break

if __name__ == "__main__":
    assert os.environ.get("VLLM_LOGGING_LEVEL",None) == "ERROR", \
        "Please set env var VLLM_LOGGING_LEVEL=ERROR"
    
    parser = ArgumentParser()
    parser.add_argument("--completions-ds", type=str, required=True)
    parser.add_argument("--mutated-ds", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--lang", choices=["py","ts"], required=True)
    parser.add_argument("--mutations", type=str, required=True)

    # dataset
    parser.add_argument("--subset", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)

    # model
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--dtype", type=str, choices=["bfloat16","float32"], default="bfloat16")

    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--max-num-candidates", type=int, default=3500)
    parser.add_argument("--seed", type=int, default=None)
    
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.overwrite:
        shutil.rmtree(Path(args.mutated_ds), ignore_errors=True)

    # check muts
    args.mutations = [m.strip() for m in args.mutations.split(',') if m != ""]
    choices = ["types","vars","delete"]
    for m in args.mutations:
        if not m in choices:
            raise NotImplementedError(f"Only accepts {choices} mutations, got {args.mutations}")
    print(f"Mutations: {args.mutations}")
    
    datasets.disable_caching()
    print("Gpu:", os.environ["CUDA_VISIBLE_DEVICES"])

    args.tokenizer=args.tokenizer if args.tokenizer else args.model
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    llm = load_vllm(args.model, args.dtype, num_available_devices(),
                    tokenizer=args.tokenizer, async_inference=True)
    model_fim = get_model_fim(args.model)
    
    ds = load_dataset(args.completions_ds, split=args.split, name=args.subset, streaming=True)
    ds = ds.to_iterable_dataset() if isinstance(ds, datasets.Dataset) else ds
    ds = ds.shuffle(seed=args.seed, buffer_size=2000)
    main(llm, tokenizer, ds, Path(args.mutated_ds), model_fim, args.batch_size, args.model, 
         args.lang, args.mutations, args.max_num_candidates)