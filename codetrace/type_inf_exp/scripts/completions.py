import datasets
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from codetrace.utils import *
import json
from argparse import ArgumentParser
import pandas as pd
from multiprocessing import cpu_count

parser = ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--prompt-ds", type=str, required=True)
parser.add_argument("--new-ds-name", type=str, required=True)
parser.add_argument("--max-size", type=int, default=1000)

args = parser.parse_args()
dataset = args.prompt_ds
model = args.model
new_name = args.new_ds_name

ds = datasets.load_dataset(dataset, split="train")
tokenizer = AutoTokenizer.from_pretrained(model)

def _condition(x):
    single_tok = (len(tokenizer.tokenize(x["fim_type"])) == 1)
    return len(x["content"]) > 1000 and len(x["content"]) < 8000 and single_tok

ds = ds.filter(_condition, num_proc=cpu_count())
# sample
ds = ds.shuffle(seed=42).select(range(args.max_size))

params = SamplingParams(temperature=0)

llm = LLM(model)
tokenizer = AutoTokenizer.from_pretrained(model)

prompts = [placeholder_to_std_fmt(ex["fim_program"], STARCODER_FIM) for ex in ds]

generations = llm.generate(prompts, params)

completions = []

for i,output in enumerate(generations):
    generated_text = output.outputs[0].text.strip()
    completions.append({**ds[i], "generated_text": generated_text, "correct": generated_text == ds[i]["fim_type"].strip(),
                        "overfull": len(tokenizer.tokenize(generated_text)) > 1,
                        "model" : "starcoderbase-1b"})
    
new_ds = datasets.Dataset.from_pandas(pd.DataFrame(completions))
new_ds.push_to_hub(new_name)
