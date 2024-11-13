import pandas as pd
import datasets
import torch
from collections import namedtuple
from copy import deepcopy
from transformers import AutoModelForCausalLM
import re
import einops
from typing import List,Union,Callable
import functools
import os
from pathlib import Path
import torch

def get_reduction(reduction:Union[Callable,str])->Callable:
    if isinstance(reduction,str):
        return getattr(torch, reduction)
    else:
        return reduction

def apply_reduction(tensor:torch.Tensor, reduction:Union[Callable,str], **kwargs)->torch.Tensor:
    reduction = get_reduction(reduction)
    x = reduction(tensor, **kwargs)
    if hasattr(x, "values"):
        return x.values
    else:
        return x
    
def get_lm_layers(model):
    if hasattr(model, "transformer"):
        return model.transformer.h
    elif hasattr(model, "model"):
        return model.model.layers
    else:
        raise NotImplementedError("Model type not supported")
    
def get_lm_final_norm(model):
    if hasattr(model, "transformer"):
        return model.transformer.ln_f
    elif hasattr(model, "model"):
        return model.model.norm
    else:
        raise NotImplementedError("Model type not supported")

def get_lm_head(model):
    if hasattr(model, "transformer") or hasattr(model, "model"):
        return model.lm_head
    else:
        raise NotImplementedError("Model type not supported")
    
def get_vllm_config(llm):
    if hasattr(llm, "llm_engine"):
        return llm.llm_engine.get_model_config().hf_config
    else:
        return llm.get_model_config().hf_config

def num_available_devices():
    device_list = list(os.environ["CUDA_VISIBLE_DEVICES"])
    return len([i for i in device_list if i != ","])

def load(ds: str, split:str=None, **kwargs) -> datasets.Dataset:
    if ds.endswith(".csv"):
        ds = datasets.Dataset.from_csv(ds,**kwargs)
    elif os.path.exists(ds):
        ds = datasets.load_from_disk(ds,**kwargs)
    else:
        ds = datasets.load_dataset(ds,**kwargs)
    return ds[split] if split else ds

def save(ds: datasets.Dataset, path:Union[str,Path], **kwargs):
    if isinstance(path, Path):
        ds.save_to_disk(str(path), **kwargs)
    else:
        ds.push_to_hub(path, **kwargs)

def lm_decode(model, x : torch.Tensor, do_norm: bool) -> torch.Tensor:
    if do_norm:
        x = get_lm_final_norm(model)(x)
    return get_lm_head(model)(x)

def masked_fill(src: torch.Tensor, mask:torch.BoolTensor, patch:torch.Tensor) -> torch.Tensor:
    # mask shape is [n_prompt, n_tok], change to [n_prompt, n_tok, dim]
    if mask.shape != src.shape:
        mask = einops.repeat(mask, "p t -> p t d", d=src.shape[-1])
    if not (src.shape == mask.shape and mask.shape == patch.shape):
        raise ValueError(f"Found different shapes: src {src.shape}, mask {mask.shape}, patch {patch.shape}")
    
    return torch.mul(src, ~mask) + torch.mul(mask, patch)

def masked_add(src: torch.Tensor, mask:torch.BoolTensor, patch:torch.Tensor) -> torch.Tensor:
    # mask shape is [n_prompt, n_tok], change to [n_prompt, n_tok, dim]
    if mask.shape != src.shape:
        mask = einops.repeat(mask, "p t -> p t d", d=src.shape[-1])
    if not (src.shape == mask.shape and mask.shape == patch.shape):
        raise ValueError(f"Found different shapes: src {src.shape}, mask {mask.shape}, patch {patch.shape}")
    
    return src + torch.mul(mask, patch)

def masked_get(src: torch.Tensor, mask:torch.BoolTensor) -> torch.Tensor:
    # mask shape is [n_prompt, n_tok], change to [n_prompt, n_tok, dim]
    if mask.shape != src.shape:
        mask = einops.repeat(mask, "p t -> p t d", d=src.shape[-1])
    if not (src.shape == mask.shape):
        raise ValueError(f"Found different shapes: src {src.shape}, mask {mask.shape}")
    return torch.mul(src, mask)

def mask_target_tokens(
    input_ids: torch.Tensor, #shape[n_prompts, n_toks, dim]
    token_ids: Union[torch.Tensor, list[int]],
    **kwargs
) -> torch.BoolTensor:
    token_ids = torch.Tensor(token_ids)
    device = kwargs.pop("device", None)
    if token_ids.ndim == 0:
        target = input_ids == token_ids.item()
    else:
        mask = functools.reduce(lambda a,b: a|b, [input_ids == i for i in token_ids])
        target = mask > 0
    
    if device:
        target = target.to(device)
    return target

def mask_target_idx(
    input_ids: torch.Tensor, #shape[n_prompts, n_toks, dim]
    indices: List[int],
    **kwargs
) -> torch.BoolTensor:
    indices = torch.Tensor(indices).to(dtype=torch.int64)
    mask = torch.zeros_like(input_ids).index_fill(1, indices, 1)
    return mask > 0

def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int,
    top_p: float,
    do_log_probs: bool
) -> torch.Tensor:
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    """
    if top_k > 0:
        if do_log_probs:
            topk_indices = logits.log_softmax(dim=-1).topk(top_k, dim=-1).indices
        else:
            topk_indices = logits.softmax(dim=-1).topk(top_k, dim=-1).indices
        # keep only indices that are in the top_k
        logits = torch.gather(logits, -1, topk_indices)
        sorted_indices = topk_indices

    if top_p < 1.0:
        raise NotImplementedError("use top_k only for now, top_p not needed for greedy decoding")
    
    TopkTuple = namedtuple('TopkTuple', ['indices','values'])
    logit_tuple = TopkTuple(indices=sorted_indices, values=logits)
    return logit_tuple

def predict(model, tokenizer, prompts: Union[List[str],str])->List[str]:
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    model.eval()
    with torch.no_grad():
        outputs = model.forward(**inputs)
        # All items in batch, last token in sequence, all logits
    last_token_id_logits = outputs.logits[:, -1, :]
    last_token_id_dists = torch.softmax(last_token_id_logits, dim=1)
    last_token_ids = torch.argmax(last_token_id_dists, dim=1)
    last_token_ids = last_token_ids.to("cpu")
    last_tokens = [ tokenizer.decode(token) for token in last_token_ids ]
    return last_tokens

def copy_decoder(modelname:str, dtype:torch.dtype) -> torch.nn.Module:
    """
    Make a copy of the model's decoder on cpu
    """
    model = AutoModelForCausalLM.from_pretrained(modelname).to("cpu", dtype=dtype)
    decoder = deepcopy(model.lm_head)
    norm = deepcopy(model.transformer.ln_f)
    del model
    decoder = torch.nn.Sequential(norm, decoder).to("cpu")
    return decoder

def keep_columns(ds: datasets.Dataset, cols:List[str]) -> datasets.Dataset:
    columns = [c for c in ds.column_names if c not in cols]
    return ds.remove_columns(columns)

def dedup_ds_by_key(ds: datasets.Dataset, key:str) -> datasets.Dataset:
    """
    Dedup ds by key. Picks the first occurence of key.
    """
    seen = set()
    new_ds = []
    for x in ds:
        if not x[key] in seen:
            new_ds.append(x)
            seen.add(x[key])
    return datasets.Dataset.from_pandas(pd.DataFrame(new_ds))

def pos_indexing(x: int, n: int) -> int:
    """
    Given a possibly negative index X over range N,
    turn to a positive index
    """
    return x if x >= 0 else n + x
