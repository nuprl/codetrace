import datasets
from vllm import LLM, SamplingParams
from codetrace.utils import *
import json
from argparse import ArgumentParser
import pandas as pd

parser = ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--prompt-ds", type=str, required=True)
parser.add_argument("--new-ds-name", type=str, required=True)

args = parser.parse_args()
dataset = args.prompt_ds
model = args.model
new_name = args.new_ds_name

ds = datasets.load_dataset(dataset, split="train")

params = SamplingParams(temperature=0)

LLM = LLM(model)

prompts = [placeholder_to_std_fmt(ex["fim_program"], STARCODER_FIM) for ex in ds]

generations = LLM.generate(prompts, params)

completions = []

for i,output in enumerate(generations):
    generated_text = output.outputs[0].text.strip()
    completions.append({**ds[i], "generated_text": generated_text, "correct": generated_text == ds[i]["fim_type"].strip()})
    
new_ds = datasets.Dataset.from_pandas(pd.DataFrame(completions))
new_ds.push_to_hub(new_name)
