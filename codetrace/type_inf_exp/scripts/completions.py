import datasets
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from codetrace.parsing_utils import get_model_fim, placeholder_to_std_fmt
import json
from argparse import ArgumentParser
import pandas as pd
from multiprocessing import cpu_count
from tqdm import tqdm
from codetrace.utils import num_available_devices
from collections import Counter
from codetrace.fast_utils import get_batches_fast, batched_do_func
import torch
import os

# filter by 1 token answer
def filter_1tok(batch, tokenizer):
    new_batch = []
    for b in batch:
        if len(tokenizer(b["fim_type"], add_special_tokens=False)["input_ids"]) == 1:
            new_batch.append(b)
    return new_batch

def main(args):
    """
    NOTE: completions are 1 token. A completion is correct if it matches the type annotation exactly.
    Thus, fim_type must be 1 token.
    """
    dataset = args.prompt_ds
    model = args.model
    new_name = args.new_ds_name
    tokenizer_name=args.tokenizer if args.tokenizer else model
    os.makedirs(new_name, exist_ok=True)

    # get model basename
    model_name = args.model_name if args.model_name else os.path.basename(model)
    print(f"Model: {model_name}")

    ds = datasets.load_dataset(dataset, split=args.split)
    ds = ds.shuffle()

    params = SamplingParams(temperature=0, max_tokens=1)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    llm = LLM(model, dtype=args.dtype, tensor_parallel_size=num_available_devices(), tokenizer=tokenizer_name)

    batches = get_batches_fast(ds, cpu_count())
    data = batched_do_func(batches, cpu_count(), filter_1tok, tokenizer=tokenizer)
    def yielder():
        for ex in tqdm(data, desc="Yielding", total=len(data)):
            yield ex
        
    ds = datasets.Dataset.from_generator(yielder)
    if args.max_size > -1:
        ds = ds.select(range(args.max_size))

    prompts = [placeholder_to_std_fmt(ex["fim_program"], get_model_fim(args.model)) for ex in ds]
                    
    completions = []
    num_correct=0
    num_processed=0
    if len(prompts) > 10000:
        print("Doing batch generations")
        batch_size = 1000
        # batch generations because of cpu ops in vllm
        for n,i in tqdm(enumerate(range(0, len(prompts), batch_size)), desc="Batch generations", total=len(prompts)//batch_size):
            generations = llm.generate(prompts[i:i+batch_size], params, use_tqdm=False)

            for j,output in enumerate(generations):
                generated_text = output.outputs[0].text.strip()
                correct = generated_text == ds[i+j]["fim_type"].strip()
                completions.append({
                    **ds[i+j], 
                    "generated_text": generated_text, 
                    "correct": correct,
                    "model" : model_name
                })
                num_processed +=1
                if correct:
                    num_correct += 1
                    print(f"Num correct {num_correct} / {num_processed} = {num_correct/num_processed}")
                
            if n % 10 == 0 and n > 0:
                # save every n batches
                print(f"Saving {n}th batch")
                new_ds = datasets.Dataset.from_pandas(pd.DataFrame(completions))
                new_ds.save_to_disk(new_name)

    else:
        generations = llm.generate(prompts, params)

        for i,output in enumerate(generations):
            generated_text = output.outputs[0].text.strip()
            correct = generated_text == ds[i]["fim_type"].strip()
            completions.append({
                **ds[i], 
                "generated_text": generated_text, 
                "correct": correct,
                "model" : model_name
            })
            num_processed +=1
            if correct:
                num_correct += 1
                print(f"Num correct {num_correct} / {num_processed} = {num_correct/num_processed}")
        
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
    parser.add_argument("--split",type=str,default="train")
    parser.add_argument("--dtype", choices=[torch.bfloat16, torch.float32], default=torch.bfloat16)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--tokenizer", default=None)
    args = parser.parse_args()
    main(args)