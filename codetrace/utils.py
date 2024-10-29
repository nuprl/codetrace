import pandas as pd
import datasets
import tree_sitter
from tree_sitter import Language, Parser
import torch
from collections import namedtuple
from pathlib import Path
from typing import List, Union
from copy import deepcopy
from transformers import AutoModelForCausalLM
import re

REPO_ROOT = Path(__file__).parent.parent
Language.build_library(
    f"{REPO_ROOT}/build/my-languages.so",
    [f"{REPO_ROOT}/tree-sitter-typescript/typescript",f"{REPO_ROOT}/tree-sitter-python"],
)
TS_LANGUAGE = Language(f"{REPO_ROOT}/build/my-languages.so", "typescript")
TS_PARSER = Parser()
TS_PARSER.set_language(TS_LANGUAGE)

PY_LANGUAGE = Language(f"{REPO_ROOT}/build/my-languages.so", "python")
PY_PARSER = Parser()
PY_PARSER.set_language(PY_LANGUAGE)

lang_to_parser = {"typescript" : TS_PARSER, "python" : PY_PARSER, "py" : PY_PARSER, "ts" : TS_PARSER}
lang_to_builder = {"typescript" : TS_LANGUAGE, "python" : PY_LANGUAGE, "py" : PY_LANGUAGE, "ts" : TS_LANGUAGE}

class FimObj:
    def __init__(
        self,
        fim_prefix : str,
        fim_suffix : str,
        fim_token : str,
        fim_placeholder : str
    ):
        self.prefix = fim_prefix
        self.suffix = fim_suffix
        self.token = fim_token
        self.placeholder = fim_placeholder
        
    def __str__(self):
        return f"FimObj(prefix={self.prefix}, suffix={self.suffix}, token={self.token}, placeholder={self.placeholder})"
    
    def to_list(self):
        return [self.prefix, self.suffix, self.token, self.placeholder]
    
fim_placeholder = "<FILL>"      
STARCODER_FIM = FimObj("<fim_prefix>", "<fim_suffix>","<fim_middle>", fim_placeholder)

def placeholder_to_std_fmt(prompt : str, fim : FimObj) -> str:
    """
    Take a prompt in fill format and convert it to standard format
    
    >>> "def func(n : <FILL>)"
    "<prefix>def func(n : <suffix>)<fim_token>"
    """
    parts = prompt.split(fim.placeholder)
    if len(parts) != 2:
        raise ValueError(f"Prompt does not contain a fim placeholder: {fim.placeholder}")
    prompt = fim.prefix + parts[0] + fim.suffix + parts[1] + fim.token
    return prompt

def std_to_placeholder_fmt(prompt : str, fim : FimObj) -> str:
    """
    Take a prompt in standard format and convert it to fill format
    
    >>> "<prefix>def func(n : <suffix>)<fim_token>"
    "def func(n : <FILL>)"
    
    """
    new_prompt = prompt.replace(fim.prefix, "").replace(fim.suffix, fim.placeholder).replace(fim.token,"")
    if fim.placeholder not in new_prompt:
        raise ValueError(f"Prompt does not contain a fim placeholder: {fim.placeholder}")
    return new_prompt

def unfim(text : str, fim : FimObj) -> str:
    """
    Remove fim special tokens and unscramble
    """
    prefix = text.split(fim.prefix)[-1].split(fim.suffix)[0]
    suffix = text.split(fim.suffix)[-1].split(fim.token)[0]
    middle = text.split(fim.token)[-1]
    return prefix+middle+suffix

def get_captures(
    prompt : Union[str,tree_sitter.Tree, bytes], 
    query: Union[str, tree_sitter.binding.Query],
    language : str = "typescript"
) -> List[tree_sitter.Node]:
    """
    Get captures for a prompt given a query
    Ignores any captures whose parents match some pattern in ignore_parents
    """
    parser = lang_to_parser[language]
    lang = lang_to_builder[language]
    if isinstance(prompt, str):
        tree = parser.parse(bytes( prompt, "utf8"))
    elif isinstance(prompt, tree_sitter.Tree):
        tree = prompt
    elif isinstance(prompt, bytes):
        tree = parser.parse(prompt)
    
    if isinstance(query, str):
        query = lang.query(query)
        
    captures = query.captures(tree.root_node)
    return captures

def replace_between_bytes(
    text : Union[str,bytes],
    start_byte : int, 
    end_byte : int,
    replacement : Union[str,bytes] = ""
) -> bytes:
    '''
    Replace tree-sitter interval (start_point, end_point) from a string.
    Inclusive of start_point and end_point
    '''
    if isinstance(replacement, str):
        replacement = replacement.encode("utf-8")
    if isinstance(text, str):
        text = text.encode("utf-8")
        
    modified_byte_string = (
        text[:start_byte] + replacement + text[end_byte:]
    )
    return modified_byte_string

def find_between_bytes(
    text : Union[str,bytes],
    start_byte : int, 
    end_byte : int,
    target : Union[str,bytes]
) -> int:
    '''
    Find the first occurence of target between start_byte and end_byte
    '''
    if isinstance(target, str):
        target = bytes(target, "utf-8")
    if isinstance(text, str):
        text = bytes(text, "utf-8")
        
    for i in range(start_byte, end_byte):
        if text[i:i+len(target)] == target:
            return i
    return -1

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

def get_next_tokens(model, tokenizer, prompts):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    model.eval()
    with torch.no_grad():
        outputs = model.forward(**inputs)
        # All items in batch, last token in sequence, all logits
    last_token_id_logits = outputs.logits[:, -1, :]
    last_token_id_dists = torch.softmax(last_token_id_logits, dim=1)
    last_token_ids = torch.argmax(last_token_id_dists, dim=1)
    last_token_ids = last_token_ids.to("cpu").tolist()
    last_tokens = [ tokenizer.decode(token) for token in last_token_ids ]
    return last_tokens

def make_decoder_copy(modelname:str) -> torch.nn.Module:
    """
    Make a copy of the model's decoder on cpu
    """
    model = AutoModelForCausalLM.from_pretrained(modelname).to("cpu")
    decoder = deepcopy(model.lm_head)
    norm = deepcopy(model.transformer.ln_f)
    del model
    decoder = torch.nn.Sequential(norm, decoder).to("cpu")
    return decoder

def keep_columns(ds, cols):
    columns = [c for c in ds.column_names if c not in cols]
    return ds.remove_columns(columns)

def dedup_ds_by_key(ds, key):
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


typescript_builtin_objects = [
    "globalThis",
    "Infinity",
    "NaN",
    "undefined",
    "Object",
    "Function",
    "Boolean",
    "Symbol",
    "Error",
    "AggregateError",
    "EvalError",
    "RangeError",
    "ReferenceError",
    "SyntaxError",
    "TypeError",
    "URIError",
    "InternalError",
    "Number",
    "BigInt",
    "Math",
    "Date",
    "String",
    "RegExp",
    "Array",
    "Int8Array",
    "Uint8Array",
    "Uint8ClampedArray",
    "Int16Array",
    "Uint16Array",
    "Int32Array",
    "Uint32Array",
    "BigInt64Array",
    "BigUint64Array",
    "Float32Array",
    "Float64Array",
    "Map",
    "Set",
    "WeakMap",
    "WeakSet",
    "ArrayBuffer",
    "SharedArrayBuffer",
    "DataView",
    "Atomics",
    "JSON",
    "WeakRef",
    "FinalizationRegistry",
    "Iterator",
    "AsyncIterator",
    "Promise",
    "GeneratorFunction",
    "AsyncGeneratorFunction",
    "Generator",
    "AsyncGenerator",
    "AsyncFunction",
    "Reflect",
    "Proxy",
    "Intl",
    "Intl.Collator",
    "Intl.DateTimeFormat",
    "Intl.DisplayNames",
    "Intl.DurationFormat",
    "Intl.ListFormat",
    "Intl.Locale",
    "Intl.NumberFormat",
    "Intl.PluralRules",
    "Intl.RelativeTimeFormat",
    "Intl.Segmenter",
    "bigint"
]

def parse_callable(tokens_to_patch):
    if not isinstance(tokens_to_patch, str):
        # ignore lists, int
        return tokens_to_patch
    
    base_module = ".".join(tokens_to_patch.split(".")[:-1])
    fn = tokens_to_patch.split(".")[-1]
    
    if base_module == "":
        try:
            # return callable from local namespace
            module = importlib.import_module(__name__)
            return getattr(module, fn)
        except:
            return tokens_to_patch
    else:
        try:
            # import and return callable from module
            module = importlib.import_module(base_module)
            return getattr(module, fn)
        except:
            return tokens_to_patch
    
def test_parse_callable():
    assert parse_callable("codetrace.fast_utils.get_batches_fast") == get_batches_fast
    assert callable(parse_callable("codetrace.fast_utils.get_batches_fast"))
    assert parse_callable(5) == 5
    assert parse_callable([1,2,3]) == [1,2,3]
    assert parse_callable(["hi","bye"]) == ["hi","bye"]
    assert parse_callable("hi") == "hi"
    assert parse_callable("run_layer_ablation") == run_layer_ablation
    assert callable(parse_callable("run_layer_ablation"))
    
def last_assert_statement_index(prompt_input_ids: List[int], **kwargs) -> List[int]:
    """
    Find the index of the function call in a prompt
    """
    tokenizer = kwargs.get("tokenizer")
    prompt = tokenizer.decode(prompt_input_ids)
    captures = re.findall(f"(assert\s+\w+\s*\()", prompt)
    # assert only one func def
    assert len(captures) == 1
    func_call_statement = captures[0]
    tokenized = tokenizer.encode(func_call_statement)
    # find the last occurence of the sublist tokenized in prompt_input_ids
    last_idx = -1
    for i in range(len(prompt_input_ids)):
        if list(prompt_input_ids[i:i+len(tokenized)]) == list(tokenized):
            last_idx = i
    return list(range(last_idx, last_idx+len(tokenized)))