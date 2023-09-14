import torch
import re
import gc
from transformers import StoppingCriteria

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
Model generation utils
"""
def layername(model, num, kind=None):
        if hasattr(model, "transformer"):
            if kind == "embed":
                return "transformer.wte"
            return f'transformer.h.{num}{"" if kind is None else "." + kind}'
        if hasattr(model, "gpt_neox"):
            if kind == "embed":
                return "gpt_neox.embed_in"
            if kind == "attn":
                kind = "attention"
            return f'gpt_neox.layers.{num}{"" if kind is None else "." + kind}'
        assert False, "unknown transformer structure"    
 

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

def trace_decode(model_out, do_greedy_decoding, gather_only_topk_logits):
    ## last layer logits     
    final_logits, past_key_values = model_out.logits, model_out.past_key_values
    softmax_out = torch.nn.functional.softmax(final_logits[:, -1, :], dim=1)

    # Top-k gathering
    topk_logits = torch.topk(softmax_out, gather_only_topk_logits, dim=1).indices
    softmax_out_top_k = torch.gather(softmax_out, 1, topk_logits)
    softmax_out_top_k = softmax_out_top_k / softmax_out_top_k.sum(1)[:, None]

    if not do_greedy_decoding:
        ## top-k sample from multinomial distribution
        new_tok_indices = torch.multinomial(softmax_out_top_k, 1)
        new_toks = torch.gather(topk_logits, 1, new_tok_indices)
    else:
        ## greedy decoding
        new_tok_indices = torch.topk(softmax_out_top_k, dim=1, k=1)
        new_toks = torch.gather(topk_logits, 1, new_tok_indices.indices)

    return new_toks, topk_logits, softmax_out_top_k

"""
StoppingCritera
"""

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops = [], encounters=None):
        super().__init__()
        if encounters is None:
            self.encounters = [1]*len(stops)
        else:
            self.encounters=encounters
        self.stops = [stop[0].tolist() for stop in stops]
        self.stop_count = [i-1 for i in self.encounters]
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        in_ids = input_ids[0].tolist()
        for i,stop in enumerate(self.stops):
            ## count occurence of sublist stop, eg. [44 54] in input_ids eg. [1 2 3 44 54]
            
            if(in_ids[-len(stop):] == stop):
                self.stop_count[i] += 1
                
        if any([self.stop_count[j] >= self.encounters[j] for j in range(len(self.stop_count))]):
            return True
        return False