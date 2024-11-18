import datasets
from vllm import LLM, SamplingParams
from codetrace.parsing_utils import get_model_fim, FimObj, FimChat
from argparse import ArgumentParser
from multiprocessing import cpu_count
from tqdm import tqdm
from codetrace.utils import num_available_devices, get_vllm_config, request_vllm_generations
from codetrace.fast_utils import make_batches, batched_apply
import torch
from typing import List,Dict,Any,Union, Optional
import os

def _success_rate(ds)->str:
    df = ds.to_pandas()
    num_succ = df["correct"].sum()
    num_tot = df["correct"].count()
    mean = df["correct"].mean()*100
    return f"Success rate: {num_succ}/{num_tot} = {mean:.2f} %"

# filter by 1 token answer
def filter_1tok(batch:List[str], tokenizer) -> List[str]:
    new_batch = []
    for b in batch:
        if len(tokenizer(b["fim_type"], add_special_tokens=False)["input_ids"]) == 1:
            new_batch.append(b)
    return new_batch

def save_data_for_resume(ds:datasets.Dataset, path:str, seed:int):
    ds.save_to_disk(path)
    with open(f"{path}/seed.md", "w") as fp:
        fp.write(seed)

def try_resume_completions(path:str, seed:int) -> List[Dict[str,Any]]:
    if os.path.exists(f"{path}/seed.md"):
        with open(f"{path}/seed.md", "r") as fp:
            saved_seed = fp.read()
        if saved_seed == seed:
            return list(datasets.load_from_disk(path))
        
    return []

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
    ds: datasets.Dataset,
    new_ds_name:str,
    model_fim: Union[FimObj,FimChat],
    max_n: Optional[int] = None,
    seed: Optional[int] = None
):
    """
    NOTE: completions are 1 token. A completion is correct if it matches the type annotation exactly.
    Thus, fim_type must be 1 token.
    """
    # filter 1 tok
    batches = make_batches(ds, cpu_count())
    data = batched_apply(batches, cpu_count(), filter_1tok, tokenizer=llm.get_tokenizer())
    ds = datasets.Dataset.from_list(data)

    # generate                  
    # batch generations because of cpu ops in vllm
    completions = try_resume_completions(new_ds_name, seed)
    ds = ds.select(range(len(completions), max_n))

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
            
            if batch_index > 0:
                # save every batch
                print(f"Saving {batch_index}th batch")
                new_ds = datasets.Dataset.from_list(completions)
                save_data_for_resume(new_ds, new_ds_name, seed)
                print(_success_rate(new_ds))

    new_ds = datasets.Dataset.from_list(completions)
    save_data_for_resume(new_ds, new_ds_name, seed)
    print(new_ds)
    print(_success_rate(new_ds))

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

    # Seed + new_ds_name will be used for resumption
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    args.tokenizer=args.tokenizer if args.tokenizer else args.model

    ds = datasets.load_dataset(args.prompt_ds, split=args.split).shuffle(args.seed)
    llm = LLM(args.model, dtype=args.dtype, tensor_parallel_size=num_available_devices(), tokenizer=args.tokenizer)
    model_fim = get_model_fim(args.model)
    max_n = len(ds) if args.max_size < 0 else args.max_size
    main(llm, ds, model_fim, args.new_ds_name, max_n)