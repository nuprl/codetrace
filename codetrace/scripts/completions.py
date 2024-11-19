import datasets
from vllm import LLM, SamplingParams
from codetrace.parsing_utils import get_model_fim, FimObj, FimChat
from argparse import ArgumentParser
from multiprocessing import cpu_count
from tqdm import tqdm
from codetrace.utils import (
    num_available_devices,
    get_vllm_config,
    request_vllm_generations,
    hex_encode,
    filter_with_blacklist
)
import torch
from typing import List,Dict,Any,Union, Optional, Set
import os
from transformers import AutoTokenizer

def success_rate(ds: datasets.Dataset )->str:
    df = ds.to_pandas()
    num_succ = df["correct"].sum()
    num_tot = df["correct"].count()
    mean = df["correct"].mean()*100
    return f"Success rate: {num_succ}/{num_tot} = {mean:.2f} %"

def _filter_1tok(batch: List[Dict[str,Any]], blacklist_key:str, sha256_blacklist: Set[Any], tokenizer):
    filtered = []
    for item in batch:
        if (not hex_encode(item[blacklist_key]) in sha256_blacklist) and \
            len(tokenizer(item["fim_type"], add_special_tokens=False)["input_ids"]) == 1:
            filtered.append(item)
    return filtered

def generate_completions(
    llm:LLM,
    prompts:Union[List[str], List[List[Dict[str,str]]]],
    ds:datasets.Dataset,
    use_tqdm:bool,
) -> List[Dict[str,Any]]:
    params = SamplingParams(temperature=0, max_tokens=1)
    model_name = get_vllm_config(llm).name_or_path
    generations = request_vllm_generations(llm, prompts, params, use_tqdm=use_tqdm)
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

def main(
    llm: LLM,
    tokenizer,
    ds: datasets.Dataset,
    new_ds_name:str,
    model_fim: Union[FimObj,FimChat],
    max_n: Optional[int] = None,
    overwrite: Optional[bool] = None
):
    """
    NOTE: completions are 1 token. A completion is correct if it matches the type annotation exactly.
    Thus, fim_type must be 1 token.
    """
    # resume from completions if they exist
    completions = []
    if not overwrite and os.path.exists(new_ds_name):
        completions = datasets.load_from_disk(new_ds_name)

    ds = filter_with_blacklist(ds, completions, "fim_program", filter_fn=_filter_1tok, tokenizer=tokenizer,
                               desc="Resuming from saved completions")
    completions = list(completions)

    if max_n > -1:
        ds = ds.select(range(max_n))
    print(ds)

    # generate                  
    # batch generations because of cpu ops in vllm
    if len(ds) < 10000:
        prompts = [model_fim.placeholder_to_fim(ex["fim_program"]) for ex in ds]
        completions += generate_completions(llm,prompts,ds, True)
    else:
        print("Doing batch generations")
        batch_size = 1000
        
        for batch_index, ds_idx in tqdm(
            enumerate(range(0,len(ds), batch_size)), 
            desc="Batch generations", 
            total=len(ds)//batch_size
        ):
            ds_batch = ds.select(range(ds_idx, min(ds_idx+batch_size, len(ds))))
            batch_prompts = [model_fim.placeholder_to_fim(ex["fim_program"]) for ex in ds_batch]
            use_tqdm = True if batch_index == 0 else False
            batch_completions = generate_completions(llm,batch_prompts,ds_batch,use_tqdm)
            completions += batch_completions
            
            # save every batch
            print(f"Saving {batch_index}th batch")
            new_ds = datasets.Dataset.from_list(completions)
            new_ds.save_to_disk(new_ds_name)
            print(success_rate(new_ds))

    new_ds = datasets.Dataset.from_list(completions)
    new_ds.save_to_disk(new_ds_name)
    print(new_ds)
    print(success_rate(new_ds))

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

    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    args.tokenizer=args.tokenizer if args.tokenizer else args.model
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    ds = datasets.load_dataset(args.prompt_ds, split=args.split).shuffle(args.seed)
    llm = LLM(args.model, dtype=args.dtype, tensor_parallel_size=num_available_devices(), tokenizer=args.tokenizer)
    model_fim = get_model_fim(args.model)

    main(llm, tokenizer, ds, args.new_ds_name, model_fim, args.max_size, args.overwrite)