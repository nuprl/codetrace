import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
from baukit import nethook
import warnings
import gc
import unicodedata
from typing import Optional, List
import collections
import numpy as np


def check_dev(n):
    t = torch.cuda.get_device_properties(n).total_memory
    r = torch.cuda.memory_reserved(n)
    a = torch.cuda.memory_allocated(n)
    f = r-a  # free
    print(f"{a} / {t} used for device {n}, reserved {r}")
    
def check_devs():
    for i in range(torch.cuda.device_count()):
        check_dev(i)
        
# may not work
def clear_devs():
    gc.collect()
    torch.cuda.empty_cache()
    
def untuple(x):
    if isinstance(x, tuple):
        return x[0]
    return x


def print_formatted_results(prompts, txt, ret_dict):
    for i in range(len(prompts)):
        print(prompts[i])
        print(txt[i])
        if('answer' in ret_dict):
            answer = ret_dict['answer'][i]['candidates']
            print("p(answer): ", ", ".join([f"p(\'{t['token']}\'[{t['token_id']}])={t['p']}" for t in answer]))
        if('p_interesting_words' in ret_dict):
            p_interesting = ret_dict['p_interesting_words'][i]
            print("p(interesting): ", ", ".join([f"p(\'{t['token']}\'[{t['token_id']}])={t['p']}" for t in p_interesting]))

        print()

        
def extract_layer_formats(named_params_iterator):
    mlp = None
    attn = None
    layers = None
    for n,p in named_params_iterator():
        n = n.split(".")
        if mlp and attn and layers:
            break
        elif "mlp" in n:
            layer = re.sub('\d+', '{}', ".".join(n[:n.index("mlp")]))
            mlp = re.sub('\d+', '{}', ".".join(n[:n.index("mlp")+1]))
        elif "attn" in n:
            attn = re.sub('\d+', '{}', ".".join(n[:n.index("attn")+1]))
        
    return {"mlp":mlp, "attn":attn, "layer":layer}
