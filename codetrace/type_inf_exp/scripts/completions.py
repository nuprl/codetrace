import datasets
from vllm import LLM, SamplingParams
from codetrace.utils import *
import json
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--prompt-ds", type=str, required=True)
parser.add_argument("--output-json", type=str, required=True)

args = parser.parse_args()
dataset = args.prompt_ds
model = args.model
output_json = args.output_json

ds = datasets.load_dataset(dataset, split="train")

params = SamplingParams(temperature=0)

LLM = LLM(model)

prompts = [placeholder_to_std_fmt("# Predict the correct type.\n"+ex["fim_program"], STARCODER_FIM) for ex in ds]
hexshas = [ex["hexsha"] for ex in ds]
solutions = [ex["fim_type"] for ex in ds]

generations = LLM.generate(prompts, params)

completions = {"generated" : [], "solution" : solutions, "hexsha" : hexshas, "prompt" : prompts}

for i,output in enumerate(generations):
    generated_text = output.outputs[0].text.strip()
    completions["generated"].append(generated_text)
    
with open(output_json, "w") as f:
    json.dump(completions, f)
    
