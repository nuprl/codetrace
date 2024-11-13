import datasets
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from codetrace.parsing_utils import get_model_fim
import json
from argparse import ArgumentParser
import pandas as pd
from multiprocessing import cpu_count
from tqdm import tqdm
from codetrace.utils import num_available_devices, get_vllm_config
from collections import Counter
from codetrace.fast_utils import make_batches, batched_apply
import torch
from typing import List,Dict,Any
import os

# filter by 1 token answer
def filter_1tok(batch:List[str], tokenizer) -> List[str]:
    new_batch = []
    for b in batch:
        if len(tokenizer(b["fim_type"], add_special_tokens=False)["input_ids"]) == 1:
            new_batch.append(b)
    return new_batch

def generate_completions(
    llm:LLM,
    prompts:List[str],
    params:SamplingParams,
    ds:datasets.Dataset
) -> List[Dict[str,Any]]:
    model_name = get_vllm_config(llm).name_or_path
    generations = llm.generate(prompts, params, use_tqdm=False)
    completions = []
    for i,output in enumerate(generations):
        generated_text = output.outputs[0].text.strip()
        correct = generated_text == ds[i]["fim_type"].strip()
        completions.append({
            **ds[i], 
            "generated_text": generated_text, 
            "correct": correct,
            "model" : model_name
        })
    return completions

def main(args):
    """
    NOTE: completions are 1 token. A completion is correct if it matches the type annotation exactly.
    Thus, fim_type must be 1 token.
    """
    args.tokenizer=args.tokenizer if args.tokenizer else args.model
    os.makedirs(args.new_ds_name, exist_ok=True)

    ds = datasets.load_dataset(args.prompt_ds, split=args.split)
    ds = ds.shuffle()

    params = SamplingParams(temperature=0, max_tokens=1)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    llm = LLM(args.model, dtype=args.dtype, tensor_parallel_size=num_available_devices(), tokenizer=tokenizer)

    # filter 1 tok
    batches = make_batches(ds, cpu_count())
    data = batched_apply(batches, cpu_count(), filter_1tok, tokenizer=tokenizer)
    def yielder():
        for ex in tqdm(data, desc="Yielding", total=len(data)):
            yield ex
        
    ds = datasets.Dataset.from_generator(yielder)
    if args.max_size > -1:
        ds = ds.select(range(args.max_size))

    # generate
    model_fim = get_model_fim(args.model)                    
    
    if len(prompts) < 10000:
        prompts = [model_fim.placeholder_to_fim(ex["fim_program"]) for ex in ds]
        completions = generate_completions(llm,prompts,params,ds)
    else:
        # batch generations because of cpu ops in vllm
        print("Doing batch generations")
        completions = []
        batch_size = 1000
        
        for batch_index, ds_idx in tqdm(enumerate(range(0,len(ds), batch_size)), desc="Batch generations", total=len(prompts)//batch_size):
            ds_batch = ds.select(range(ds_idx, min(ds_idx+batch_size, len(ds))))
            batch_prompts = [model_fim.placeholder_to_fim(ex["fim_program"]) for ex in ds_batch]
            batch_completions = generate_completions(llm,batch_prompts,params,ds_batch)
            completions += batch_completions
            
            if batch_index % 10 == 0 and batch_index > 0:
                # save every n batches
                print(f"Saving {batch_index}th batch")
                new_ds = datasets.Dataset.from_list(completions)
                new_ds.save_to_disk(args.new_ds_name)
                print(Counter(new_ds["correct"]))

    new_ds = datasets.Dataset.from_list(completions)
    print(new_ds)
    new_ds.save_to_disk(args.new_ds_name)
    print(Counter(new_ds["correct"]))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--prompt-ds", type=str, required=True)
    parser.add_argument("--new-ds-name", type=str, required=True)
    parser.add_argument("--max-size", type=int, default=-1)
    parser.add_argument("--split",type=str,default="train")
    parser.add_argument("--dtype", choices=[torch.bfloat16, torch.float32], default=torch.bfloat16)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--tokenizer", default=None)
    args = parser.parse_args()
    main(args)