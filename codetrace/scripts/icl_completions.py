import shutil
from argparse import ArgumentParser
from pathlib import Path
import asyncio
from typing import List,Dict,Any,Union,Dict
import os
import torch
import datasets
from vllm import AsyncLLMEngine
from codetrace.parsing_utils import get_model_fim, FimObj, FimChat, prepare_fim_prompt
from transformers import AutoTokenizer, PreTrainedTokenizer
from tqdm import tqdm
from codetrace.utils import (
    num_available_devices,
    hex_encode,
    load_dataset
)
from codetrace.vllm_utils import (
    generate_completions,
    load_vllm
)

def prepare_icl_prompt(context_dataset: datasets.Dataset, fim_program:str, model:str) -> str:
    model_fim = get_model_fim(model)
    fim_program = model_fim.placeholder_to_fim(fim_program)
    context = context_dataset.shuffle().select(range(2))
    context = [model_fim.placeholder_to_fim(x["mutated_program"])+x["fim_type"] for x in context]
    return "\n\n".join(context + [fim_program])

def success_rate(ds: datasets.Dataset) -> str:
    df = ds.to_pandas()
    num_succ = df["correct"].sum()
    num_tot = df["correct"].count()
    mean = df["correct"].mean()*100
    return f"Success rate: {num_succ}/{num_tot} = {mean:.2f} %"

def is_1tok(fim_type: str, tokenizer: PreTrainedTokenizer) -> bool:
    return len(tokenizer(fim_type, add_special_tokens=False)["input_ids"]) == 1

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
    print(success_rate(new_ds))

def main(
    llm: AsyncLLMEngine,
    ds: datasets.IterableDataset,
    new_ds_path: Path,
    batch_size: int,
    model_name: str,
    max_n: int
):
    # resume from completions if they exist
    completions, blacklist = [], set()
    if os.path.exists(new_ds_path):
        completions = datasets.load_from_disk(new_ds_path, keep_in_memory=False)
        print(f"Resuming from {len(completions)} completions.")
        for row in completions:
            blacklist.add(hex_encode(row["fim_program"]))

    # preprocess dataset
    if len(blacklist) > 0:
        ds = ds.filter(lambda x: hex_encode(x["fim_program"]) not in blacklist)
    
    # generate                  
    # batch generations because of cpu ops in vllm
    num_completed = 0
    for i,batch in tqdm(enumerate(ds.iter(batch_size)), desc="Batch generations"):
        icl_prompt = batch["_prompt"]
        batch_completions = asyncio.run(generate_completions(
                                        llm,
                                        batch,
                                        batch_size,
                                        use_tqdm=(i == 0)
                                    ))
        batch_completions = [{**x,
                              "icl_prompt":icl_prompt[b],
                              "generated_text": x["_generated"],
                              "correct": x["_generated"] == x["fim_type"], 
                              "model_name": model_name} for b,x in enumerate(batch_completions)]
        num_completed += len(batch_completions)
        # save every batch
        _save(batch_completions, new_ds_path, f"Saving {i} batch")
        if max_n > 0 and num_completed >= max_n:
            break

if __name__ == "__main__":
    assert os.environ.get("VLLM_LOGGING_LEVEL",None) == "ERROR", \
        "Please set env var VLLM_LOGGING_LEVEL=ERROR"
    
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--prompt-ds", type=str, required=True)
    parser.add_argument("--subset", type=str, default=None)
    parser.add_argument("--new-ds-name", type=str, required=True)
    
    parser.add_argument("--max-size", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=1000)

    parser.add_argument("--dtype", choices=[torch.bfloat16, torch.float32], default=torch.bfloat16)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--tokenizer", default=None)

    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    if args.overwrite:
        shutil.rmtree(Path(args.new_ds_name))

    args.tokenizer=args.tokenizer if args.tokenizer else args.model
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    datasets.disable_caching()
    if Path(args.prompt_ds).exists():
        ds_prompts = load_dataset(f"{args.prompt_ds}/test-0-of-1.parquet")
        ds_icl_context = load_dataset(f"{args.prompt_ds}/steer-0-of-1.parquet")
    else:
        ds = datasets.load_dataset(args.prompt_ds, name=args.subset)
        ds_prompts = ds["test"]
        ds_icl_context = ds["steer"]
    
    ds = ds_prompts.map(
        lambda x: {**x, "_prompt": prepare_icl_prompt(ds_icl_context, x["mutated_program"], args.model)}
    )
    llm = load_vllm(args.model, args.dtype, num_available_devices(),
                    tokenizer=args.tokenizer, async_inference=True)
    model_fim = get_model_fim(args.model)
    
    main(llm, ds, Path(args.new_ds_name), args.batch_size,args.model, args.max_size)