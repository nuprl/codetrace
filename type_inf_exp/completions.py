import datasets
from vllm import LLM, SamplingParams
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import *
import json

ds = datasets.load_dataset("franlucc/stenotype-eval-dataset-func-type-stripped-v4", split="train")

params = SamplingParams(temperature=0)

model = "/home/arjun/models/starcoderbase-1b"
LLM = LLM(model)

prompts = [placeholder_to_std_fmt("# Predict the correct type.\n"+ex["fim_program"], STARCODER_FIM) for ex in ds]
hexshas = [ex["hexsha"] for ex in ds]
solutions = [ex["fim_type"] for ex in ds]

generations = LLM.generate(prompts, params)

completions = {"generated" : [], "solution" : solutions, "hexsha" : hexshas, "prompt" : prompts}

for i,output in enumerate(generations):
    generated_text = output.outputs[0].text.strip()
    completions["generated"].append(generated_text)
    
with open("starcoderbase-3b-completions-cheeky.json", "w") as f:
    json.dump(completions, f)
    
