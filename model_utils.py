import torch
import re
import warnings
import gc
from text_generation import Client
from concurrent.futures.thread import ThreadPoolExecutor
from tqdm import tqdm
import transformers
from typing import List, Tuple, Optional
from transformers import StoppingCriteriaList, StoppingCriteria, AutoConfig
from transformer_lens import HookedTransformerConfig

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
    

def code_print(generated_list, line_numbers=True):
    txt = "".join(generated_list)
    if line_numbers:
        for n, i in enumerate(txt.rstrip().split('\n')):
            print(n, i)
    else:
        print(txt)

        {
  "_name_or_path": "bigcode/santacoder",
  "activation_function": "gelu_fast",
  "architectures": [
    "GPT2LMHeadCustomModel"
  ],
  "attention_head_type": "multiquery",
  "attn_pdrop": 0.1,
  "auto_map": {
    "AutoConfig": "bigcode/santacoder--configuration_gpt2_mq.GPT2CustomConfig",
    "AutoModelForCausalLM": "bigcode/santacoder--modeling_gpt2_mq.GPT2LMHeadCustomModel"
  },
  "bos_token_id": 49152,
  "embd_pdrop": 0.1,
  "eos_token_id": 49152,
  "initializer_range": 0.02,
  "layer_norm_epsilon": 1e-05,
  "model_type": "gpt2",
  "n_embd": 2048,
  "n_head": 16,
  "n_inner": 8192,
  "n_layer": 24,
  "n_positions": 2048,
  "reorder_and_upcast_attn": false,
  "resid_pdrop": 0.1,
  "scale_attn_by_inverse_layer_idx": false,
  "scale_attn_weights": true,
  "summary_activation": null,
  "summary_first_dropout": 0.1,
  "summary_proj_to_labels": true,
  "summary_type": "cls_index",
  "summary_use_proj": true,
  "torch_dtype": "float32",
  "transformers_version": "4.30.0.dev0",
  "use_cache": true,
  "vocab_size": 49280
}

# d_model (int): The dimensionality of the embeddings.
# d_head (int): The dimensionality of each attention head.
# n_layers (int): The number of transformer blocks (one block = one attn layer AND one MLP layer).
# n_ctx (int): The maximum sequence length.
# n_heads (int): The number of attention heads.
        
hooked_to_bigcode = {
    "n_layers": "n_layer",
    "d_model": "n_embd",
    "n_ctx": "n_ctx",
    "d_head": "n_head",
    "model_name": "_name_or_path",
    "n_heads": "n_head",
#     d_mlp: Optional[int] = None
    "act_fn": "activation_function",
#     d_vocab: int = -1
    "eps" : "layer_norm_epsilon",
#     use_attn_result: bool = False
#     "use_attn_scale": "scale_attn_weights" TODO????,
#     use_split_qkv_input: bool = False
#     use_local_attn: bool = False  CACHE???
#     "original_architecture": "architectures", TODO: architectures is a list?
#     from_checkpoint: bool = False
#     checkpoint_index: Optional[int] = None
#     checkpoint_label_type: Optional[str] = None
#     checkpoint_value: Optional[int] = None
    "tokenizer_name": "_name_or_path",
#     window_size: Optional[int] = None
#     "attn_types": "attention_head_type", TODO: list to : str?
#     init_mode: str = "gpt2"
#     normalization_type: Optional[str] = "LN"
#     device: Optional[str] = None
#     n_devices: int = 1
#     attention_dir: str = "causal"
#     attn_only: bool = False
#     seed: Optional[int] = None
#     initializer_range: float = -1.0
#     init_weights: bool = True
    # "scale_attn_by_inverse_layer_idx": "scale_attn_by_inverse_layer_idx",
#     positional_embedding_type: str = "standard"
#     final_rms: bool = False
#     d_vocab_out: int = -1
#     parallel_attn_mlp: bool = False
#     rotary_dim: Optional[int] = None
#     n_params: Optional[int] = None
#     use_hook_tokens: bool = False
#     gated_mlp: bool = False
}


        
def bigcode_to_hooked_config(config):
    
    hooked_config_params = {}
    
    # d_model (int): The dimensionality of the embeddings.
    # d_head (int): The dimensionality of each attention head.
    
    for hooked_key in hooked_to_bigcode.keys():
        if hooked_key == "n_ctx":
            hooked_config_params[hooked_key] = 512 # max seq len
        else:
            hooked_config_params[hooked_key] = getattr(config, hooked_to_bigcode[hooked_key])
    
    hooked_config = HookedTransformerConfig(**hooked_config_params)
    return hooked_config
    
    