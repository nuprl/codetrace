import shutil
from argparse import ArgumentParser
from pathlib import Path
import asyncio
from typing import List,Dict,Any,Union,Dict
import os
import torch
import datasets
from vllm import AsyncLLMEngine
from codetrace.parsing_utils import get_model_fim, FimObj, FimChat
from transformers import AutoTokenizer, PreTrainedTokenizer
from tqdm import tqdm
import itertools as it
from codetrace.utils import (
    num_available_devices,
    hex_encode,
    load_dataset
)
from codetrace.parsing_utils import LLAMA_CHAT_TEMPLATE, LLAMA_FIM_CHAT
from codetrace.vllm_utils import (
    generate_completions,
    load_vllm
)

def _add_to_prompt(prompt: Union[str, List[Dict[str,str]]], label:str)->Union[str, List[Dict[str,str]]]:
    if isinstance(prompt, str):
        return prompt + label
    else:
        agent_dict = prompt[-1]
        agent_dict["content"] = agent_dict["content"] + label
        prompt[-1] = agent_dict
        return prompt
    
def _join_prompts(prompts: List[Union[str, List[Dict[str,str]]]]) -> Union[str, List[Dict[str,str]]]:
    if isinstance(prompts[0], str):
        return "\n\n".join(prompts)
    else:
        return list(it.chain(*prompts))

def _prepare_prompt(
    tokenizer: PreTrainedTokenizer, 
    model_fim: Union[FimObj,FimChat], 
    prompt: Union[str,List[Dict[str,str]]]
) -> str:
    if isinstance(model_fim, FimChat):
        chat_template = LLAMA_CHAT_TEMPLATE \
                        if model_fim == LLAMA_FIM_CHAT \
                        else tokenizer.get_chat_template()
        
        return tokenizer.apply_chat_template(
            prompt, 
            tokenize=False, 
            add_generation_prompt=False,
            continue_final_message=True,
            chat_template=chat_template
        )
    else:
        return prompt
    
def prepare_icl_prompt(context_dataset: datasets.Dataset, fim_program:str, model:str, tokenizer:PreTrainedTokenizer) -> str:
    datasets.disable_progress_bars()
    model_fim = get_model_fim(model)
    fim_program = model_fim.placeholder_to_fim(fim_program)
    context = context_dataset.shuffle().filter(lambda x: len(x["mutated_program"]) < 1000).select(range(2))
    context = [_add_to_prompt(model_fim.placeholder_to_fim(x["mutated_program"]),x["fim_type"]) for x in context]
    return _prepare_prompt(tokenizer, model_fim, _join_prompts(context + [fim_program]))

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
    # save final success rate
    final_succ = success_rate(datasets.Dataset.from_list(batch_completions))
    with open(new_ds_path / "success.md","w") as fp:
        fp.write(final_succ)

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
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    ds = ds_prompts.map(
        lambda x: {**x, "_prompt": prepare_icl_prompt(ds_icl_context, x["mutated_program"], args.model, tokenizer)}
    )
    llm = load_vllm(args.model, args.dtype, num_available_devices(),
                    tokenizer=args.tokenizer, async_inference=True)
    model_fim = get_model_fim(args.model)
    args.batch_size = min(args.batch_size,len(ds))
    main(llm, ds, Path(args.new_ds_name), args.batch_size,args.model, args.max_size)


"""
Tests
"""

def test_prompts():
    import json
    fim_prompt = """
def palindrome(s: <FILL>):
    return s[::-1]==s""".strip()

    label = "str"
    model_fim = get_model_fim("starcoderbase-1b")
    chat_model_fim = get_model_fim("codellama-7b-instruct-hf")
    output_fim = _join_prompts([_add_to_prompt(model_fim.placeholder_to_fim(fim_prompt), label),
                                model_fim.placeholder_to_fim(fim_prompt)])
    output_chat = _join_prompts([_add_to_prompt(chat_model_fim.placeholder_to_fim(fim_prompt), label),
                                 chat_model_fim.placeholder_to_fim(fim_prompt)])
    expected_fim = """
<fim_prefix>def palindrome(s: <fim_suffix>):
    return s[::-1]==s<fim_middle>str

<fim_prefix>def palindrome(s: <fim_suffix>):
    return s[::-1]==s<fim_middle>""".strip()
    expected_chat = [
        {"role": "user", "content": """Continue this program with the correct substitution for <FILL>:

def palindrome(s: <FILL>):
    return s[::-1]==s"""},
        {"role": "assistant", "content": '''def palindrome(s: str'''},
        {"role": "user", "content": """Continue this program with the correct substitution for <FILL>:

def palindrome(s: <FILL>):
    return s[::-1]==s"""},
        {"role": "assistant", "content": '''def palindrome(s: '''}

    ]
    print(json.dumps(output_chat, indent=4), "\n\n", json.dumps(expected_chat, indent=4))
    assert output_fim == expected_fim
    assert output_chat == expected_chat