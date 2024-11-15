import tree_sitter
import tree_sitter_python as tspython
import tree_sitter_typescript as tstypescript
from tree_sitter import Language, Parser
from typing import List,Union,Dict,Tuple

PY_LANGUAGE = Language(tspython.language())
PY_PARSER = Parser(PY_LANGUAGE)
TS_LANGUAGE = Language(tstypescript.language_typescript())
TS_PARSER = Parser(TS_LANGUAGE)

lang_to_parser = {"typescript" : TS_PARSER, "python" : PY_PARSER, "py" : PY_PARSER, "ts" : TS_PARSER}
lang_to_builder = {"typescript" : TS_LANGUAGE, "python" : PY_LANGUAGE, "py" : PY_LANGUAGE, "ts" : TS_LANGUAGE}

class FimObj:
    def __init__(
        self,
        fim_prefix : str,
        fim_suffix : str,
        fim_middle : str,
        fim_placeholder : str,
        token_ids: Dict[str, int]
    ):
        self.prefix = fim_prefix
        self.suffix = fim_suffix
        self.middle = fim_middle
        self.placeholder = fim_placeholder
        self.token_ids = token_ids
    
    def __str__(self):
        return f"FimObj(prefix={self.prefix}, suffix={self.suffix}, middle={self.middle}"
    
    def get_token_ids(self, key:str):
        return self.token_ids[key]
    
    def to_list(self):
        return [self.prefix, self.suffix, self.middle, self.placeholder]
    
    def placeholder_to_fim(self, prompt:str)->str:
        assert self.is_placeholder(prompt) and not self.is_fim(prompt), \
            f"Prompt is not in fim placeholder format: {self.placeholder}!"
        parts = prompt.split(self.placeholder)
        prompt = self.prefix + parts[0] + self.suffix + parts[1] + self.middle
        return prompt
    
    def fim_to_placeholder(self, prompt:str)->str:
        assert self.is_fim(prompt) and not self.is_placeholder(prompt), "Prompt is not in fim format!"
        return prompt.replace(self.prefix, "").replace(self.suffix, self.placeholder).replace(self.middle,"")

    def unfim(self, prompt:str) -> str:
        assert self.is_fim(prompt) and not self.is_placeholder(prompt), "Prompt is not in fim format!"
        prefix = prompt.split(self.prefix)[-1].split(self.suffix)[0]
        suffix = prompt.split(self.suffix)[-1].split(self.middle)[0]
        middle = prompt.split(self.middle)[-1]
        return prefix+middle+suffix
    
    def is_placeholder(self, prompt:str)->bool:
        return self.placeholder in prompt
    
    def is_fim(self, prompt:str)->bool:
        return all([t in prompt for t in [self.prefix, self.suffix, self.middle]])
    
fim_placeholder = "<FILL>"      
STARCODER_FIM = FimObj("<fim_prefix>","<fim_suffix>","<fim_middle>",fim_placeholder,
                       {"<fim_prefix>":1, "<fim_suffix>":3,"<fim_middle>":2})

# https://github.com/gonglinyuan/safim/blob/main/model_utils.py
CODELLAMA_FIM = FimObj("<PRE>", " <SUF>"," <MID>", fim_placeholder,
                       {"<PRE>":32007, " <SUF>":32008," <MID>":32009}) #TODO: why the space??

# note the order changes in deepseek
# https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-base
DEEPSEEK_FIM = FimObj("<｜fim▁begin｜>", "<｜fim▁hole｜>", "<｜fim▁end｜>", fim_placeholder,
                      {"<｜fim▁begin｜>":32016, "<｜fim▁hole｜>":32015,"<｜fim▁end｜>":32017})

# Based on https://huggingface.co/Qwen/Qwen2.5-Coder-7B/blob/main/tokenizer_config.json
QWEN_FIM = FimObj(
    "<|fim_prefix|>",
    "<|fim_suffix|>",
    "<|fim_middle|>",
    fim_placeholder,
    { "<|fim_prefix|>": 151659, "<|fim_suffix|>": 151661, "<|fim_middle|>": 151660 }
)

def get_model_fim(model_name:str) -> FimObj:
    if "starcoder" in model_name.lower():
        return STARCODER_FIM
    elif "codellama" in model_name.lower():
        return CODELLAMA_FIM
    elif "deepseek" in model_name.lower():
        return DEEPSEEK_FIM
    elif "qwen" in model_name.lower():
        return QWEN_FIM
    else:
        raise NotImplementedError(f"Not supported FIM model: {model_name}")

def get_captures(
    prompt : Union[str,tree_sitter.Tree, bytes], 
    query: Union[str, tree_sitter.Query],
    language : str,
    key: str
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
    
    query = lang.query(query) if isinstance(query, str) else query
        
    captures = query.captures(tree.root_node)
    if captures != {}:
        return captures[key]
    else:
        return []

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
    modified_byte_string = (text[:start_byte] + replacement + text[end_byte:])
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

def is_in_capture_range(node : tree_sitter.Node, captures : List[Tuple[tree_sitter.Node, str]]) -> bool:
    """
    Check if the node is in the range of any of the captures.
    """
    for capture in captures:
        if node.start_byte >= capture.start_byte and node.end_byte <= capture.end_byte:
            return True
    return False

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

"""
PYTESTS
"""

def test_captures():
    prompt = "def func:"
    query="((identifier) @name)"
    captures = get_captures(prompt, query, "python", "name")
    assert len(captures) == 1, captures
    assert captures[0].text.decode("utf-8") == "func", captures

def test_fim():
    prompt = "hi my name is <FILL>! Nice to meet you"
    fim_prompt = STARCODER_FIM.placeholder_to_fim(prompt)
    unfimmed_prompt = STARCODER_FIM.unfim(fim_prompt + "George")
    placeholder_prompt = STARCODER_FIM.fim_to_placeholder(fim_prompt)
    assert placeholder_prompt == "hi my name is <FILL>! Nice to meet you"
    assert unfimmed_prompt == "hi my name is George! Nice to meet you"
    assert fim_prompt == f"{STARCODER_FIM.prefix}hi my name is {STARCODER_FIM.suffix}! Nice to meet you{STARCODER_FIM.middle}"
