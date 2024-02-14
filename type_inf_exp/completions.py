import datasets
from vllm import LLM, SamplingParams
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import *
import json

ds = datasets.load_dataset("franlucc/stenotype-eval-dataset-func-type-stripped-v3", split="train")

params = SamplingParams(temperature=0)

model = "/home/arjun/models/starcoderbase-1b"
LLM = LLM(model)

prompts = [placeholder_to_std_fmt(ex["fim_program"], STARCODER_FIM) for ex in ds if "<FILL>" in ex["fim_program"]]
solution = [ex["fim_type"].strip()[1:-1] for ex in ds]

generations = LLM.generate(prompts, params)

completions = []
success = 0
maybe_success = 0
for i,output in enumerate(generations):
    generated_text = output.outputs[0].text.strip()
    completions.append(generated_text)
    if generated_text == solution[i]:
        success += 1
    if generated_text.startswith(solution[i]):
        maybe_success += 1
    
with open("starcoderbase-1b-completions.json", "w") as f:
    json.dump(completions, f)
    
# count success
print(f"Success rate: {success/len(ds)}, total: {success}/{len(ds)}")
print(f"Maybe success rate: {maybe_success/len(ds)}, total: {maybe_success}/{len(ds)}")