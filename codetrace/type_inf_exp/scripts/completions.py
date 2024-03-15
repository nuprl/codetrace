import datasets
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from codetrace.utils import *
import json
from argparse import ArgumentParser
import pandas as pd
from multiprocessing import cpu_count
from tqdm import tqdm
from collections import Counter

parser = ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--prompt-ds", type=str, required=True)
parser.add_argument("--new-ds-name", type=str, required=True)
parser.add_argument("--max-size", type=int, default=-1)

args = parser.parse_args()
dataset = args.prompt_ds
model = args.model
new_name = args.new_ds_name

# get model basename
model_name = model.split("/")[-1]
print(f"Model: {model_name}")

ds = datasets.load_dataset(dataset, split="train")
tokenizer = AutoTokenizer.from_pretrained(model)

# sample
if args.max_size > -1:
    ds = ds.select(range(args.max_size))

params = SamplingParams(temperature=0)

llm = LLM(model)
tokenizer = AutoTokenizer.from_pretrained(model)

prompts = [placeholder_to_std_fmt(ex["fim_program"], STARCODER_FIM) for ex in ds]

completions = []
if len(prompts) > 10000:
    print("Doing batch generations")
    batch_size = 1000
    # batch generations because of RAM in vllm
    for n,i in tqdm(enumerate(range(0, len(prompts), batch_size)), desc="Batch generations"):
        generations = llm.generate(prompts[i:i+batch_size], params, use_tqdm=False)

        for j,output in enumerate(generations):
            generated_text = output.outputs[0].text.strip()
            completions.append({**ds[i+j], 
                                "generated_text": generated_text, 
                                "correct": generated_text.startswith(ds[i+j]["fim_type"].strip()),
                                "overfull": len(tokenizer.tokenize(generated_text)) > len(tokenizer.tokenize(ds[i+j]["fim_type"].strip())),
                                "model" : model_name})
            
        if n % 10 == 0 and n > 0:
            print(f"Saving {batch_size} completions")
            new_ds = datasets.Dataset.from_pandas(pd.DataFrame(completions))
            new_ds.push_to_hub(new_name)

else:
    generations = llm.generate(prompts, params)

    for i,output in enumerate(generations):
        generated_text = output.outputs[0].text.strip()
        completions.append({**ds[i], 
                            "generated_text": generated_text, 
                            "correct": generated_text.startswith(ds[i]["fim_type"].strip()),
                            "overfull": len(tokenizer.tokenize(generated_text)) > len(tokenizer.tokenize(ds[i]["fim_type"].strip())),
                            "model" : model_name})

    
new_ds = datasets.Dataset.from_pandas(pd.DataFrame(completions))
new_ds.push_to_hub(new_name)


# print some counts
print(Counter(new_ds["correct"]))
print(Counter(new_ds["overfull"]))