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
    hex_encode
)
from codetrace.vllm_utils import (
    generate_completions,
    load_vllm
)

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
    tokenizer: PreTrainedTokenizer,
    ds: datasets.IterableDataset,
    new_ds_path: Path,
    model_fim: Union[FimObj,FimChat],
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
    ds = ds.filter(lambda x: is_1tok(x["fim_type"], tokenizer))
    if len(blacklist) > 0:
        ds = ds.filter(lambda x: hex_encode(x["fim_program"]) not in blacklist)
    
    ds = ds.map(lambda x: {**x, "_prompt": prepare_fim_prompt(tokenizer, model_fim, x["fim_program"])})

    # generate                  
    # batch generations because of cpu ops in vllm
    num_completed = 0
    for i,batch in tqdm(enumerate(ds.iter(batch_size)), desc="Batch generations"):
        batch_completions = asyncio.run(generate_completions(
                                        llm,
                                        batch,
                                        batch_size,
                                        use_tqdm=(i == 0)
                                    ))
        batch_completions = [{**x, 
                              "generated_text": x["_generated"],
                              "correct": x["_generated"] == x["fim_type"], 
                              "model_name": model_name} for x in batch_completions]
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
    parser.add_argument("--new-ds-name", type=str, required=True)
    
    parser.add_argument("--max-size", type=int, default=-1)
    parser.add_argument("--split",type=str,default="train")
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
    ds = datasets.load_dataset(args.prompt_ds, split=args.split, streaming=True).shuffle(
                                                                args.seed, buffer_size=2000)
    
    llm = load_vllm(args.model, args.dtype, num_available_devices(),
                    tokenizer=args.tokenizer, async_inference=True)
    model_fim = get_model_fim(args.model)
    
    main(llm, tokenizer, ds, Path(args.new_ds_name), model_fim, args.batch_size,
         args.model, args.max_size)