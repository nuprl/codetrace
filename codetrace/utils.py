import pandas as pd
import datasets
import torch
from collections import namedtuple
from copy import deepcopy
from transformers import AutoModelForCausalLM
from typing import List,Union,Callable, Dict, Any, TypeVar, Set, Generator
from tqdm import tqdm
import os
from pathlib import Path
import torch
import functools
from hashlib import sha256
from multiprocessing import cpu_count
import einops
from torchtyping import TensorType

# Activation Tensors
HiddenStateStack = TypeVar(TensorType[float,"num_layer","num_prompt","num_tokens","hdim"])
HiddenStateStack_1tok = TypeVar(TensorType[float,"num_layer","num_prompt","hdim"])

HiddenState = TypeVar(TensorType[float, "num_prompt","num_tokens","hdim"])
HiddenState_1tok = TypeVar(TensorType[float, "num_prompt","hdim"])

InputTensor = TypeVar(TensorType[float, "n_prompt","num_tokens"])
InputMaskTensor = TypeVar(TensorType[bool, "n_prompt","num_tokens"])

MaskTensorStack = TypeVar(TensorType[bool,"num_layer","num_prompt","num_tokens","hdim"])
MaskTensor = TypeVar(TensorType[bool,"num_prompt","num_tokens","hdim"])

# Logit tensors
LogitsStack = TypeVar(TensorType[float,"num_layer","num_prompt","num_tokens","vocab_size"])
Logits = TypeVar(TensorType[float,"num_prompt","num_tokens","vocab_size"])

# Prediction Tensors
IndicesTensor = TypeVar(TensorType[int, "num_layer","num_prompt","num_tokens","top_k"])
ProbabilitiesTensor = TypeVar(TensorType[float, "num_layer","num_prompt","num_tokens","top_k"])

def hex_encode(s: str) -> str:
    return sha256(bytes(s, "utf-8")).hexdigest()

def apply_reduction(
    tensor: torch.Tensor, 
    reduction: Union[str, Callable[[torch.Tensor,int], torch.Tensor]], 
    dim: int,
    **kwargs
) -> torch.Tensor:
    """
    Applies a reduction function at given dim. Reduction fn is so called
    because it reduces the size or number of dimensions of tensor.
    Resulting tensor will always be smaller.
    """
    if isinstance(reduction, str):
        if reduction == "max":
            return tensor.amax(dim=dim,**kwargs)
        elif reduction == "sum":
            return tensor.sum(dim=dim,**kwargs)
        else:
            raise NotImplementedError("Reduction not implemented")
    else:
        return reduction(tensor,dim=dim, **kwargs)

def get_lm_hdim(model) -> int:
    if hasattr(model, "transformer") or hasattr(model, "model"):
        return model.lm_head.in_features
    else:
        raise NotImplementedError("Model type not supported")

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

def num_available_devices():
    device_list = list(os.environ["CUDA_VISIBLE_DEVICES"])
    return len([i for i in device_list if i != ","])

def load_dataset(ds: str, split:str=None, **hub_kwargs) -> datasets.Dataset:
    if ds.endswith(".csv"):
        ds = datasets.Dataset.from_csv(ds)
    elif os.path.exists(ds):
        ds = datasets.load_from_disk(ds)
    else:
        ds = datasets.load_dataset(ds, **hub_kwargs)
    return ds[split] if split else ds

def save_dataset(ds: datasets.Dataset, path:Union[str,Path], **hub_kwargs):
    if isinstance(path, Path):
        ds.save_to_disk(path.as_posix())
    else:
        ds.push_to_hub(path, **hub_kwargs)

def lm_decode(model, x : torch.Tensor, do_norm: bool) -> torch.Tensor:
    if do_norm:
        x = get_lm_final_norm(model)(x)
    return get_lm_head(model)(x)

def masked_fill(src: torch.Tensor, mask: torch.BoolTensor, patch: torch.Tensor) -> torch.Tensor:
    """
    Replaces SRC tensor with PATCH values at MASK locations. Must have same sizes.

    >>> masked_fill( [1,2,3], [1,0,0], [4,5,6])
    [4,2,3]
    """
    if not (src.shape == mask.shape and mask.shape == patch.shape):
        raise ValueError(f"Found different shapes: src {src.shape}, mask {mask.shape}, patch {patch.shape}")
    
    return torch.mul(src, ~mask) + torch.mul(mask, patch)

def masked_add(src: torch.Tensor, mask: torch.BoolTensor, patch: torch.Tensor) -> torch.Tensor:
    """
    Adds SRC tensor with PATCH values at MASK locations. Must have same sizes.
    >>> masked_add( [1,2,3], [1,0,0], [4,6,7])
    [5,2,3]
    """
    if not (src.shape == mask.shape and mask.shape == patch.shape):
        raise ValueError(f"Found different shapes: src {src.shape}, mask {mask.shape}, patch {patch.shape}")
    
    return src + torch.mul(mask, patch)

def masked_get(src: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
    """
    Zeros out SRC values at MASK locations. Must have same sizes.
    >>> masked_get( [1,2,3], [0,1,0])
    [0,2,0]
    """
    if not (src.shape == mask.shape):
        raise ValueError(f"Found different shapes: src {src.shape}, mask {mask.shape}")
    return torch.mul(src, mask)

def mask_target_tokens(input_ids: torch.Tensor, token_ids: List[int], **kwargs) -> torch.BoolTensor:
    """
    Returns a mask tensor over INPUT_IDS s.t. where MASK == 1 then the corresponding
    value in INPUT_IDS is in TOKEN_IDS list
    >>> mask_target_tokens( [1,2,3], [3, 1])
    [1,0,1]
    """
    token_ids = torch.Tensor(token_ids)
    if token_ids.ndim == 0:
        # if 1 token id, check which members in input_ids are equal
        target = input_ids == token_ids.item()
    else:
        mask = functools.reduce(lambda a,b: a|b, [input_ids == i for i in token_ids])
        target = mask > 0
    
    device = kwargs.pop("device", None)
    if device:
        target = target.to(device)
    return target

def mask_target_idx(input_ids: torch.Tensor, indices: List[int], dim:int=1) -> torch.BoolTensor:
    """
    Returns a mask tensor over INPUT_IDS s.t. where MASK == 1 then the corresponding
    index in INPUT_IDS is in INDICES list (along a certain DIM)
    >>> mask_target_tokens( [1,2,3], [2,1], 0)
    [0,1,1]
    """
    indices = torch.Tensor(indices).to(dtype=torch.int64)
    mask = torch.zeros_like(input_ids).index_fill(dim, indices, 1)
    return mask > 0

def topk_filtering(
    logits: torch.Tensor,
    top_k: int,
    do_log_probs: bool
) -> torch.Tensor:
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    """
    assert top_k > 0
    if do_log_probs:
        logits = logits.log_softmax(dim=-1)
    else:
        logits = logits.softmax(dim=-1)

    return logits.topk(top_k, dim=-1)

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

def reset_index_dim0(x: torch.Tensor, index_labels: List[int], n: int)->  torch.Tensor:
    """
    Given a tensor X and list of INDEX_LABELS (0 < i < n) for dim 0, if dim 0
    of the tensor does not match the range of N, then re-index tensor
    such that INDEX_LABELS are now indices into the tensor. Fill rest with zeros.

    >>> reset_index_dim0([0.1, 0.2, 0.3],  [3, 7, 1], 9)
    [0, 0.3, 0, 0.1, 0, 0, 0, 0.2, 0]
    """
    if n == x.shape[0]:
        return x
    
    new_shape = list(x.shape)
    new_shape[0] = n
    new_tensor = torch.zeros(new_shape, dtype=x.dtype, device=x.device)
    new_tensor[index_labels] = x
    
    return new_tensor