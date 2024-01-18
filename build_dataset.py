import glob
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from vllm import LLM, SamplingParams
import os
import json

class FimObj:
    def __init__(self,
                 fim_prefix : str,
                 fim_suffix : str,
                 fim_token : str,
                 fim_placeholder : str):
        self.prefix = fim_prefix
        self.suffix = fim_suffix
        self.token = fim_token
        self.placeholder = fim_placeholder


def placeholder_to_std_fmt(prompt : str, fim : FimObj) -> str:
    parts = prompt.split(fim.placeholder)
    prompt = fim.prefix + parts[0] + fim.suffix + parts[1] + fim.token
    return prompt

def std_to_placeholder_fmt(prompt : str, fim : FimObj) -> str:
    return prompt.replace(fim.prefix, "").replace(fim.suffix, fim.placeholder).replace(fim.token,"")


def generate_test(prompts : Tuple[str, str], 
                  llm : LLM, 
                  llm_name : str,
                  max_tokens : int, 
                  params : Dict[str, SamplingParams],
                 fim : FimObj):
    inputs = [placeholder_to_std_fmt(p[1], fim) for p in prompts]
    for k,prm in params.items():
        if max_tokens == 1:
            outdir = f"generations/{llm_name}/singletok"
        else:
            outdir = f"generations/{llm_name}/multitok"
        prm.max_tokens=max_tokens
        outputs = llm.generate(inputs, prm)
        os.makedirs(outdir, exist_ok=True)
        for i,output in enumerate(outputs):
            progname = prompts[i][0]
            generated_text = []
            for j in range(len(output.outputs)):
                text = output.outputs[j].text
                generated_text.append(text)
            with open(f"{outdir}/{progname}_{k}_{output.request_id}_n{j+1}.json", "w") as f:
                json.dump({"content":generated_text, "sampling_params": params.__repr__()}, f)


# from MultiPL-E
def estimator(n: int, c: int, k: int) -> float:
    """
    Calculates 1 - comb(n - c, k) / comb(n, k).
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

def pass_k(gens : List[str], sol : str, k : int):
    n = len(gens)
    c = len([i for i in gens if (sol == i)])
    return estimator(n, c, k)
    

def type_inf_eval(gen_dir : str, verbose=False):
    results = {}
    for f in glob.glob(f"{gen_dir}/*.json"):
        stem = Path(f).stem
        gens = json.load(open(f, "r"))["content"]
        # solution = json.load(open(f"dataset/{stem.split('_')[0]}/solutions.json", "r"))
        # sols = [v for k,v in solution.items() if k in stem]
        # if len(sols) != 1: ## hacky
        #     raise ArgumentError("There should be one solution, check your solutions.json file")
        # sol = sols[0]
        ## get sol from file name
        sol = stem.split("_fim_")[-1].split("_")[0]
        if len(gens) == 1:
            passk = (gens[0] == sol)
        else:
            passk = pass_k(gens, sol, 1)
        if verbose:
            results[stem] = [ passk, gens, sol]
        else:
            results[stem] = passk
    return results


# prompts is list of [(prog_fname, content)]
def get_prompts(targets : Optional[List[str]]=None) -> List[str]:
    """
    targets allows you to select only certain dataset programs eg. "bible" or "date"
    """
    prompts = []
    for f in glob.glob(f"dataset/*/*.ts"):
        if not targets:
            bool = True
        else:
            bool = any([t in f for t in targets])
        if bool and "fim" in f:
            prog_name = Path(f).stem
            prompts.append((prog_name, open(f, "r").read()))
    return prompts

# sampling types, note default max tok is 16
sampling_creative = SamplingParams(temperature=0.8, top_p=0.95, n=20)
sampling_eval = SamplingParams(temperature=0.2, n=20)
greedy_params = SamplingParams(temperature=0)