import torch
import re
import gc

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
    
def untuple(x):
    if isinstance(x, tuple):
        return x[0]
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

def print_formatted(prompts, generations, only_gen=True):
    for i,txt in enumerate(generations):
        if only_gen:
            out = f"## Gen {i}:\n{generations[i]}\n"
        else:
            out = f"## Prompt {i}:\n{prompts[i]}\n## Gen {i}:\n{generations[i]}\n"
        print(out)
