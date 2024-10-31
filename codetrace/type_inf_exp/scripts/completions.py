import datasets
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from codetrace.parsing_utils import STARCODER_FIM, placeholder_to_std_fmt
import json
from argparse import ArgumentParser
import pandas as pd
from multiprocessing import cpu_count
from tqdm import tqdm
from collections import Counter
from codetrace.fast_utils import get_batches_fast, batched_do_func
import torch
import os

# filter by 1 token answer
def filter_1tok(batch, tokenizer):
    new_batch = []
    for b in batch:
        if len(tokenizer.encode(b["fim_type"])) == 1:
            new_batch.append(b)
    return new_batch
    
def num_available_devices():
    device_list = list(os.environ["CUDA_VISIBLE_DEVICES"])
    return len([i for i in device_list if i != ","])

def main(args):
    """
    NOTE: completions are 1 token. A completion is correct if it matches the type annotation exactly.
    Thus, fim_type must be 1 token.
    """
    dataset = args.prompt_ds
    model = args.model
    new_name = args.new_ds_name
    os.makedirs(new_name, exist_ok=True)

    # get model basename
    model_name = model.split("/")[-1]
    print(f"Model: {model_name}")

    ds = datasets.load_dataset(dataset, split="train")
    ds = ds.shuffle()

    params = SamplingParams(temperature=0, max_tokens=1)

    llm = LLM(model, dtype=args.dtype, tensor_parallel_size=num_available_devices())
    tokenizer = AutoTokenizer.from_pretrained(model)

    batches = get_batches_fast(ds, cpu_count())
    data = batched_do_func(batches, cpu_count(), filter_1tok, tokenizer=tokenizer)
    def yielder():
        for ex in tqdm(data, desc="Yielding", total=len(data)):
            yield ex
        
    ds = datasets.Dataset.from_generator(yielder)
    if args.max_size > -1:
        ds = ds.select(range(args.max_size))

    prompts = [placeholder_to_std_fmt(ex["fim_program"], STARCODER_FIM) for ex in ds]
                    
    completions = []
    if len(prompts) > 10000:
        print("Doing batch generations")
        batch_size = 1000
        # batch generations because of cpu ops in vllm
        for n,i in tqdm(enumerate(range(0, len(prompts), batch_size)), desc="Batch generations", total=len(prompts)//batch_size):
            generations = llm.generate(prompts[i:i+batch_size], params, use_tqdm=False)

            for j,output in enumerate(generations):
                generated_text = output.outputs[0].text.strip()
                completions.append({
                    **ds[i+j], 
                    "generated_text": generated_text, 
                    "correct": generated_text == ds[i+j]["fim_type"].strip(),
                    "model" : model_name
                })
                
            if n % 50 == 0 and n > 0:
                # save every n batches
                print(f"Saving {n}th batch")
                new_ds = datasets.Dataset.from_pandas(pd.DataFrame(completions))
                new_ds.save_to_disk(new_name)

    else:
        generations = llm.generate(prompts, params)

        for i,output in enumerate(generations):
            generated_text = output.outputs[0].text.strip()
            completions.append({
                **ds[i], 
                "generated_text": generated_text, 
                "correct": generated_text == ds[i]["fim_type"].strip(),
                "model" : model_name
            })

        
    new_ds = datasets.Dataset.from_pandas(pd.DataFrame(completions))
    print(new_ds)
    new_ds.save_to_disk(new_name)
    print(Counter(new_ds["correct"]))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--prompt-ds", type=str, required=True)
    parser.add_argument("--new-ds-name", type=str, required=True)
    parser.add_argument("--max-size", type=int, default=-1)
    parser.add_argument("--dtype", choices=[torch.bfloat16, torch.float32], default=torch.bfloat16)
    args = parser.parse_args()
    main(args)