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
            layer = re.sub('\d', '{}', ".".join(n[:n.index("mlp")]))
            mlp = re.sub('\d', '{}', ".".join(n[:n.index("mlp")+1]))
        elif "attn" in n:
            attn = re.sub('\d', '{}', ".".join(n[:n.index("attn")+1]))
        
    return {"mlp":mlp, "attn":attn, "layer":layer}

    
# def save_model_dict(outfn):
#     """
#     extracts a bunch of highly used fields from different model configurations
#     model_type, no_split_module_classes, layer_name_format, mlp_module_name_format, attn_module_name_format
#     """
#     model_dict = {
        
#         "bigcode/santacoder" : 
#         {
#             "model_type" : "santacoder",
#             "model_size" : "1.1B",
#             "no_split_module_classes" : ["GPT2CustomBlock"],
#             "layer_name_format" : "",
#             "mlp_module_name_format" : "",
#             "attn_module_name_format" : ""
#         },
#         "Salesforce/codegen-16B-mono" : 
#         {
#             "model_type" : "codegen",
#             "model_size" : "16B",
#             "no_split_module_classes" : ["CodeGenBlock"],
#             "layer_name_format" : "MY_PLACEHOLDER",
#             "mlp_module_name_format" : "MY_PLACEHOLDER",
#             "attn_module_name_format" : "MY_PLACEHOLDER"
#         },
        
#     }

#     model_type = None
    
#     # if(hasattr(self.model, "transformer")):
#     #     model_type = "gpt2"
#     #     no_split_module_classes = ["GPT2Block"]
#     # elif(hasattr(self.model, "gpt_neox")):
#     #     model_type = "gpt-neox"
#     #     no_split_module_classes = ["GPTNeoXLayer"]
#     # elif("llama" in config._name_or_path):
#     #     model_type = "llama"
#     #     no_split_module_classes = ["LlamaDecoderLayer"]
#     # elif("galactica" in config._name_or_path):
#     #     model_type = "galactica"
#     #     no_split_module_classes  = ["OPTDecoderLayer"]
#     # elif("codegen" in config._name_or_path):
#     #     model_type = "codegen"
#     #     no_split_module_classes  = ["CodeGenBlock"]
#     # else:
#     #     warnings.warn("unknown model type >> unable to extract relavent fields from config")

#     self.n_layer = None
#     self.n_embd = None
#     self.n_attn_head = None
#     self.max_seq_length = None

#     self.layer_name_format = None
#     self.layer_names = None
#     self.mlp_module_name_format = None
#     self.attn_module_name_format = None
#     self.ln_f_name = None
#     self.unembedder_name = None
#     self.embedder_name = None

#     self.model_type = model_type
#     self.no_split_module_classes = no_split_module_classes

# #         if(model_type in ["llama", "galactica"]):
# #             self.n_layer = config.num_hidden_layers
# #             self.n_embd = config.hidden_size
# #             self.n_attn_head = config.num_attention_heads
# #             self.max_seq_length = config.max_sequence_length

# #             layer_name_prefix = "model"
# #             if(model_type == "galactica"):
# #                 layer_name_prefix = "model.decoder"

# #             self.layer_name_format = layer_name_prefix + ".layers.{}"

# #             self.embedder_name = "model.embed_tokens"
# #             self.ln_f_name = "model.norm" if model_type=="llama" else "model.decoder.final_layer_norm"
# #             self.unembedder_name = "lm_head"

# #             if(model_type == "llama"):
# #                 self.mlp_module_name_format = "model.layers.{}.mlp"
# #             else:
# #                 self.mlp_module_name_format = "model.layers.{}.fc2" # this is the output of mlp in galactica. the input is on model.layers.{}.fc1
# #             self.attn_module_name_format = "model.layers.{}.self_attn"

# #         elif(model_type in ["gpt2", "gpt-neox"]):
# #             self.n_layer = config.n_layer
# #             self.n_embd = config.n_embd
# #             self.n_attn_head = config.n_head
# #             self.max_seq_length = config.n_ctx

# #             self.layer_name_format = "transformer.h.{}"
# #             self.embedder_name = "transformer.wte"
# #             self.ln_f_name = "transformer.ln_f"
# #             self.unembedder_name = "lm_head"
# #             self.mlp_module_name_format = "transformer.h.{}.mlp"
# #             self.attn_module_name_format = "transformer.h.{}.attn"

#     # print("num_layers >> ", self.num_layers)
#     if(model_type is not None):
#         self.layer_names = [self.layer_name_format.format(i) for i in range(self.n_layer)]
#         self.mlp_module_names = [self.mlp_module_name_format.format(i) for i in range(self.n_layer)]
#         self.attn_module_names = [self.attn_module_name_format.format(i) for i in range(self.n_layer)]
#         self.tracable_modules =  self.mlp_module_names + self.attn_module_names + self.layer_names
            