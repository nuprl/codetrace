from tqdm import tqdm
from codetrace.interp_utils import (
    collect_hidden_states, 
    logit_lens, 
    collect_hidden_states_at_tokens, 
    insert_patch,
    LogitResult,
    TraceResult
)
from codetrace.utils import make_decoder_copy
from nnsight import LanguageModel
from nnsight.tracing import Proxy
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from argparse import ArgumentParser

def test_logit_pipeline():
    prompts = [
        'print(f',
        'a=0\nb=1\nc=',
    ]
    hs = collect_hidden_states(model, prompts)

    assert hs.shape[0] == 24, hs.shape # layers
    assert hs.shape[1] == 2, hs.shape # prompt len
    # 2 is token count, padded, skip it.
    assert hs.shape[3] == model.config.n_embd, hs.shape
    
    logits = logit_lens(model, prompts)
    logits : LogitResult = logits.decode_logits( layers=[0,23], prompt_idx=[0,1])
    tok_a_f = logits[1][0].tokens(model.tokenizer)
    tok_b_f = logits[1][-1].tokens(model.tokenizer)
    assert tok_b_f == '2', tok_b_f
    assert tok_a_f == '"', tok_a_f

def patch_clean_to_corrupt(model, clean, corrupt, tokens):
    hs = collect_hidden_states_at_tokens(model, clean, tokens)
    res = insert_patch(model, corrupt, hs, list(range(len(model.transformer.h))), tokens, patch_mode="subst")
    return res

def test_patch():
    prompts = [
        'print("hi bobby!")\nprint(f',
        'a=0\nb=1\nc=',
    ]
    len_a = len(model.tokenizer.encode(prompts[0]))
    len_b = len(model.tokenizer.encode(prompts[1]))
    assert len_a == len_b, f"Change input prompts st. token length matches: {len_a} != {len_b}"
    trace_res = patch_clean_to_corrupt(model, prompts[0], prompts[1], list(range(len_a)))
    tok_pred = trace_res.decode_logits().tokens(model.tokenizer)
    assert tok_pred == '"', tok_pred
    trace_res = patch_clean_to_corrupt(model, prompts[1], prompts[0], list(range(len_a)))
    tok_pred = trace_res.decode_logits().tokens(model.tokenizer)
    assert tok_pred == '2', tok_pred
    

def test_logit_generation_match():
    prompts = [
        'print(f',
        'a=0\nb=1\nc=',
    ]
    logits = logit_lens(model, prompts)
    logits : LogitResult = logits.decode_logits(prompt_idx=[0,1])
    tok_a_f = logits[0][0].tokens(model.tokenizer)[0]
    tok_b_f = logits[0][1].tokens(model.tokenizer)[0]
    
    with model.generate(max_new_tokens=1) as gen:
        with gen.invoke(prompts) as invoker:
            invoker.next()

    output = gen.output
    toks = [model.tokenizer.decode(x) for x in output[:,-1]]
    assert toks[0] == tok_a_f, f"{toks[0]} != {tok_a_f}"
    assert toks[1] == tok_b_f, f"{toks[1]} != {tok_b_f}"
    
    
def test_collect_at_token_idx():
    prompts = [
        '<fim_prefix>print("hi"<fim_suffix>\n<fim_middle>',
        "<fim_prefix>a=6\nb=6\nc=<fim_suffix><fim_middle>",
    ]
    logits = logit_lens(model, prompts)
    logits : LogitResult = logits.decode_logits(prompt_idx=[0,1])

    tok_a_f = logits[-1][0][-1].tokens(model.tokenizer)
    tok_b_f = logits[-1][1][-1].tokens(model.tokenizer)
    assert tok_a_f == ')', f"{repr(tok_a_f)}"
    assert tok_b_f == '6', f"{repr(tok_b_f)}"
    
def test_collect_at_token_idx2():
    prompts = [
        '<fim_prefix>print("hello world"<fim_suffix>\n<fim_middle>',
        "<fim_prefix>a=6\nb=6\nc=<fim_suffix><fim_middle>",
    ]
    toks = ["<fim_prefix>", "<fim_suffix>", "<fim_middle>"]
    hs = collect_hidden_states_at_tokens(model, prompts, toks).to("cpu")
    decoder = make_decoder_copy(model.config.name_or_path)
    logits = decoder(hs)
    out : TraceResult = TraceResult(logits, list(range(len(model.transformer.h))), len(model.transformer.h))
    logits : LogitResult = out.decode_logits(prompt_idx=[0,1])

    tok_a_f = logits[-1][0][-1].tokens(model.tokenizer)
    tok_b_f = logits[-1][1][-1].tokens(model.tokenizer)
    assert tok_a_f == ')', f"{repr(tok_a_f)}"
    assert tok_b_f == '6', f"{repr(tok_b_f)}"
    
    
def test_collect_at_token_idx3():
    prompts = [
        '<fim_prefix>print("hello world"<fim_suffix>\n<fim_middle>',
        "<fim_prefix>a=6\nb=6\nc=<fim_suffix><fim_middle>",
    ]
    toks = [-1]
    hs = collect_hidden_states_at_tokens(model, prompts, toks).to("cpu")
    decoder = make_decoder_copy(model.config.name_or_path)
    logits = decoder(hs)
    out : TraceResult = TraceResult(logits, list(range(len(model.transformer.h))), len(model.transformer.h))
    logits : LogitResult = out.decode_logits(prompt_idx=[0,1])
    
    tok_a_f = logits[-1][0][-1].tokens(model.tokenizer)
    tok_b_f = logits[-1][1][-1].tokens(model.tokenizer)
    assert tok_a_f == ')', f"{repr(tok_a_f)}"
    assert tok_b_f == '6', f"{repr(tok_b_f)}"
    
def test_collect_at_token_idx4():
    prompts = [
        '<fim_prefix>print("hello world"<fim_suffix>\n<fim_middle>',
        "<fim_prefix>a=6\nb=6\nc=<fim_suffix><fim_middle>",
    ]
    toks = ["<fim_middle>"]
    hs = collect_hidden_states_at_tokens(model, prompts, toks).to("cpu")
    decoder = make_decoder_copy(model.config.name_or_path)
    logits = decoder(hs)
    out : TraceResult = TraceResult(logits, list(range(len(model.transformer.h))), len(model.transformer.h))
    logits : LogitResult = out.decode_logits(prompt_idx=[0,1])

    tok_a_f = logits[-1][0][-1].tokens(model.tokenizer)
    tok_b_f = logits[-1][1][-1].tokens(model.tokenizer)
    assert tok_a_f == ')', f"{repr(tok_a_f)}"
    assert tok_b_f == '6', f"{repr(tok_b_f)}"
    
def test_collect_at_token_idx5():
    prompts = [
        '<fim_prefix>print("hello world"<fim_suffix>\n<fim_middle>',
        "<fim_prefix>a=6\nb=6\nc=<fim_suffix><fim_middle>",
    ]
    toks = [0,-1]
    hs = collect_hidden_states_at_tokens(model, prompts, toks).to("cpu")
    decoder = make_decoder_copy(model.config.name_or_path)
    logits = decoder(hs)
    out : TraceResult = TraceResult(logits, list(range(len(model.transformer.h))), len(model.transformer.h))
    logits : LogitResult = out.decode_logits(prompt_idx=[0,1], token_idx=[0,-1])
    
    tok_a_f = logits[-1][0][-1].tokens(model.tokenizer)
    tok_b_f = logits[-1][1][-1].tokens(model.tokenizer)
    assert tok_a_f == ')', f"{repr(tok_a_f)}"
    assert tok_b_f == '6', f"{repr(tok_b_f)}"
    
def test_interp_patch():
    prompts = [
        '<fim_prefix>print("hello world"<fim_suffix>\n<fim_middle>',
        "<fim_prefix>a=6\nb=6\nc=<fim_suffix><fim_middle>",
    ]
    toks = ["<fim_prefix>", "<fim_suffix>", "<fim_middle>"]
    hs = collect_hidden_states_at_tokens(model, prompts[0], toks)
    patch = hs.expand(-1,len(prompts),-1,-1)
    out = insert_patch(model, prompts, patch, list(range(len(model.transformer.h))), toks, patch_mode="subst")
    out : LogitResult = out.decode_logits(prompt_idx=[0,1])
    tok_a_f = out[-1][0][-1].tokens(model.tokenizer)
    tok_b_f = out[-1][1][-1].tokens(model.tokenizer)
    assert tok_a_f == ')', f"{repr(tok_a_f)}"
    assert tok_b_f == ')', f"{repr(tok_b_f)}"
   

def test_generation():
    gold = '''from typing import List

def uhslzetzbdurzagwmhn(bwsgyuklklhw: str) -> List[int]:
""" Input to this function is a string represented multiple groups for nested parentheses separated by spaces.
For each of the group, output the deepest level of nesting of parentheses.
E.g. (()()) has maximum two levels of nesting while ((())) has three."""

    def xuuzjpbpbmjwlpktx(e):
        depth = 0
        max_depth = 0
        for c in e:
            if c == '(':
                depth += 1
                max_depth = max(max_depth, depth)
            elif c == ')':
                depth -= 1
                max_depth = max(max_depth, depth)
        return max_depth

    return [xuuzjpbpbmjwlpktx(bwsgyuklklhw)]'''
    
    prompt = '''from typing import List

def uhslzetzbdurzagwmhn(bwsgyuklklhw: str) -> List[int]:
""" Input to this function is a string represented multiple groups for nested parentheses separated by spaces.
For each of the group, output the deepest level of nesting of parentheses.
E.g. (()()) has maximum two levels of nesting while ((())) has three."""

    def xuuzjpbpbmjwlpktx(e):
        depth = 0
        max_depth = 0
        for c in e:
            if c == '(':
                depth += 1
                max_depth = max'''
    
    max_out = 512
    generated = []
    for i in tqdm(range(max_out)):
        res = logit_lens(model, prompt, -1)
        logits = res.decode_logits()
        tok = logits[-1][-1].tokens(model.tokenizer)[0]
        if tok == "<|endoftext|>":
            break
        generated.append(tok)
        prompt = prompt + tok

    assert prompt.strip() == gold.strip(), f"\n{prompt.strip()}\n\n\n{gold}"
    
    generated = []
    patch = torch.zeros(24, 1, 1, model.config.n_embd)
    for i in tqdm(range(max_out)):
        # patch nothing, this checks precision errors in code
        res = insert_patch(model, prompt, patch, list(len(model.transformer.h)), [-1], collect_hidden_states=False)
        logits = res.decode_logits()
        tok = logits[-1][-1].tokens(model.tokenizer)[0]
        if tok == "<|endoftext|>":
            break
        generated.append(tok)
        prompt = prompt + tok
    
    assert prompt.strip() == gold.strip(), f"\n{prompt.strip()}\n\n\n{gold}"

"""
CLI running code
"""
def repeat_test(func, n, **kwargs):
    for i in range(n):
        print(f"Running {func.__name__} {i+1}/{n}")
        func(**kwargs)
            
parser = ArgumentParser()
parser.add_argument("--modelname", type=str, required=True)
parser.add_argument("--num_reps", type=int, default=10)
args = parser.parse_args()
model = LanguageModel(args.modelname, device_map="cuda")

# repeating tests multiple times ensures no precision errors in code

repeat_test(test_logit_pipeline, args.num_reps)
repeat_test(test_patch, args.num_reps)
repeat_test(test_logit_generation_match, args.num_reps)
repeat_test(test_collect_at_token_idx, args.num_reps)
repeat_test(test_collect_at_token_idx2, args.num_reps)
repeat_test(test_collect_at_token_idx3, args.num_reps)
repeat_test(test_collect_at_token_idx4, args.num_reps)
repeat_test(test_collect_at_token_idx5, args.num_reps)
repeat_test(test_interp_patch, args.num_reps)
repeat_test(test_generation, 1)

print("All tests passed!")
