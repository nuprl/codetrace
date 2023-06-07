import torch
import re
import warnings
import gc
# from text_generation import Client
from concurrent.futures.thread import ThreadPoolExecutor
from tqdm import tqdm
import transformers
from typing import List, Tuple, Optional
from transformers import StoppingCriteriaList, StoppingCriteria, AutoConfig


"""
Cuda utils
"""
def check_dev(n):
    t = torch.cuda.get_device_properties(n).total_memory
    r = torch.cuda.memory_reserved(n)
    a = torch.cuda.memory_allocated(n)
    f = r-a  # free
    print(f"{a} / {t} used for device {n}, reserved {r}")
    
def check_devs():
    for i in range(torch.cuda.device_count()):
        check_dev(i)
        
# may not work with jupyter
def clear_devs():
    gc.collect()
    torch.cuda.empty_cache()
    

"""
Client generation utils
"""

def print_by_line(previous_text: str, new_text: str):
    """
    A little hack to print line-by-line in a Notebook. We receive results
    a few tokens at a time. This buffers output until a newline, so that
    we do not print partial lines.
    """
    if "\n" not in new_text:
        return
    last_newline = previous_text.rfind("\n")
    if last_newline != -1:
        print(previous_text[last_newline+1:] + new_text, end="")
    else:
        print(previous_text + new_text, end="")


def generate_by_client(prompt: str,
    client,
    max_new_tokens=512,
    stop_sequences=[ "\ndef", "\nclass", "\nif"  ],
    do_sample=False,
    echo=True):
    text = ""
    for response in client.generate_stream(prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.2,
        do_sample=do_sample,
        top_p=0.95,
        stop_sequences=stop_sequences):
        if not response.token.special:
            if echo:
                print_by_line(text, response.token.text)
            text += response.token.text
    if echo:
        print_by_line(text, "\n") # flush any remaining text
    return text

"""
Model generation utils
"""

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, tokenizer, stops = [], device="cuda", encounters=1):
        super().__init__()
        self.encounters=encounters
        self.tokenizer = tokenizer
        self.stops = [stop.to(device) for stop in stops]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        stop_count = []
        for stop in self.stops:
            stop_count.append(self.tokenizer.decode(input_ids[0]).count(self.tokenizer.decode(stop)))
            
        if any([stop_count[i] >= self.encounters[i] for i in range(len(stop_count))]):
            return True
        return False
    

def untuple(x):
    if isinstance(x, tuple):
        return x[-1]
    return x


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



def code_print(generated_list, line_numbers=True):
    txt = "".join(generated_list)
    if line_numbers:
        for n, i in enumerate(txt.rstrip().split('\n')):
            print(n, i)
    else:
        print(txt)


    