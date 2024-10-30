from tqdm import tqdm
from codetrace.interp_utils import (
    collect_hidden_states, 
    logit_lens,
    insert_patch,
    LogitResult,
    TraceResult
)
from codetrace.utils import (
    make_decoder_copy,
    masked_fill,
    masked_get,
    mask_target_tokens,
    mask_target_idx,
    predict
)
from nnsight import LanguageModel
import torch
from argparse import ArgumentParser
from functools import partial
import re
from typing import List
"""
Setup code
"""
parser = ArgumentParser()
parser.add_argument("--modelname", type=str, default="bigcode/starcoderbase-1b")
parser.add_argument("--num_reps", type=int, default=10)
args = parser.parse_args()
model = LanguageModel(args.modelname, device_map="cuda", torch_dtype=torch.bfloat16)

"""
tests
"""

def test_logit_pipeline():
    prompts = [
        'print(f',
        'a=0\nb=1\nc=',
    ]
    hs = collect_hidden_states(model, prompts)
    # generated = predict(model, model.tokenizer, prompts)
    assert hs.shape[0] == 24, hs.shape # layers
    assert hs.shape[1] == 2, hs.shape # prompt len
    # 2 is token count, padded, skip it.
    assert hs.shape[3] == model.config.n_embd, hs.shape
    
    logits = logit_lens(model, prompts)
    logits : LogitResult = logits.decode_logits(layers=[0,23], prompt_idx=[0,1], token_idx=[-1])
    tok_a_f = logits[1,0].tokens(model.tokenizer)
    tok_b_f = logits[1,-1].tokens(model.tokenizer)
    assert tok_b_f == '2', tok_b_f
    assert tok_a_f == '"', tok_a_f

def test_masked_fill():
    src = torch.Tensor([[1,2,3],[4,5,6]])
    mask = torch.BoolTensor([[True, False, True],[False,True,False]])
    patch = torch.Tensor([[10,20,30],[40,50,60]])
    res = masked_fill(src, mask, patch)
    expected = torch.Tensor(
        [[10,2,30],
         [4,50,6]]
    )
    assert torch.equal(res, expected)

    src = torch.Tensor([[1,2,3],[4,5,6]])
    mask = torch.BoolTensor([[1,1,1],[1,1,1]])
    patch = torch.Tensor([[10,20,30],[40,50,60]])
    res = masked_fill(src, mask, patch)
    expected = patch
    assert torch.equal(res, expected)

def test_masked_get():
    src = torch.Tensor([[1,2,3],[4,5,6]])
    mask = torch.BoolTensor([[True, False, True],[False,True,False]])
    res = masked_get(src, mask)
    expected = torch.Tensor(
        [[1,0,3],
         [0,5,0]]
    )
    assert torch.equal(res, expected)

def test_mask_target_token():
    tokens = ["<fim_middle>","<fim_prefix>"]
    prompts = [
        "<fim_prefix><fim_middle>x",
        "d<fim_middle>e"
    ]
    input_ids = model.tokenizer(prompts, return_tensors="pt")["input_ids"]
    assert input_ids.shape == (2, 3)
    result = mask_target_tokens(input_ids, tokens, model.tokenizer)
    expected = torch.BoolTensor([
        [True, True, False],
        [False, True, False]
    ])
    assert torch.equal(result, expected), f"result {result}, expected {expected}"

def test_mask_target_idx():
    indices = [0,2]
    prompts = [
        "a b x",
        "d c e"
    ]
    input_ids = model.tokenizer(prompts, return_tensors="pt")["input_ids"]
    assert input_ids.shape == (2, 3)
    result = mask_target_idx(input_ids, indices)
    expected = torch.BoolTensor([
        [True, False, True],
        [True, False, True],
    ])
    assert torch.equal(result, expected), f"result {result}, expected {expected}"

def patch_clean_to_corrupt(model, clean, corrupt, indices_or_tokens):
    if isinstance(indices_or_tokens[0], str):
        targets = partial(mask_target_tokens, tokens=indices_or_tokens, tokenizer=model.tokenizer)
    else:
        targets = partial(mask_target_idx, indices=indices_or_tokens)

    hs = collect_hidden_states(model, clean, target_fn=targets)
    res = insert_patch(
        model, 
        corrupt, 
        hs, 
        list(range(len(model.transformer.h))),
        target_fn=targets)
    return res

def test_patch():
    prompts = [
        'print("hi bobby!")\nprint(f',
        'a=0\nb=1\nc=',
    ]
    generated = predict(model, model.tokenizer, prompts)
    len_a = len(model.tokenizer.encode(prompts[0]))
    len_b = len(model.tokenizer.encode(prompts[1]))
    assert len_a == len_b, f"Change input prompts st. token length matches: {len_a} != {len_b}"
    trace_res = patch_clean_to_corrupt(model, prompts[0], prompts[1], list(range(len_a)))
    tok_pred = trace_res.decode_logits(prompt_idx=[0],token_idx=[-1]).tokens(model.tokenizer)
    assert tok_pred == '"', tok_pred

    trace_res = patch_clean_to_corrupt(model, prompts[1], prompts[0], list(range(len_a)))
    tok_pred = trace_res.decode_logits(prompt_idx=[0],token_idx=[-1]).tokens(model.tokenizer)
    assert tok_pred == '2', tok_pred
    

def test_logit_generation_match():
    prompts = [
        'print(f',
        'a=0\nb=1\nc=',
    ]
    logits = logit_lens(model, prompts)
    logits : LogitResult = logits.decode_logits(prompt_idx=[0,1])
    tok_a_f = logits[0,0,0].tokens(model.tokenizer)
    tok_b_f = logits[0,1,0].tokens(model.tokenizer)
    
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

    tok_a_f = logits[-1,0,-1].tokens(model.tokenizer)
    tok_b_f = logits[-1,1,-1].tokens(model.tokenizer)
    assert tok_a_f == ')', f"{repr(tok_a_f)}"
    assert tok_b_f == '6', f"{repr(tok_b_f)}"
    
def test_collect_at_token_idx2():
    prompts = [
        '<fim_prefix>print("hello world"<fim_suffix>\n<fim_middle>',
        "<fim_prefix>a=6\nb=6\nc=<fim_suffix><fim_middle>",
    ]
    toks = ["<fim_prefix>", "<fim_suffix>", "<fim_middle>"]
    target_fn = partial(mask_target_tokens, tokens=toks)
    hs = collect_hidden_states(model, prompts, get_target_ids=target_fn).to("cpu")
    decoder = make_decoder_copy(model.config.name_or_path)
    logits = decoder(hs)
    out : TraceResult = TraceResult(logits, list(range(len(model.transformer.h))), len(model.transformer.h))
    logits : LogitResult = out.decode_logits(prompt_idx=[0,1])

    tok_a_f = logits[-1,0,-1].tokens(model.tokenizer)
    tok_b_f = logits[-1,1,-1].tokens(model.tokenizer)
    assert tok_a_f == ')', f"{repr(tok_a_f)}"
    assert tok_b_f == '6', f"{repr(tok_b_f)}"
    
    
def test_collect_at_token_idx3():
    prompts = [
        '<fim_prefix>print("hello world"<fim_suffix>\n<fim_middle>',
        "<fim_prefix>a=6\nb=6\nc=<fim_suffix><fim_middle>",
    ]
    toks = [-1]
    target_fn = partial(mask_target_tokens, tokens=toks)
    hs = collect_hidden_states(model, prompts, get_target_ids=target_fn).to("cpu")
    decoder = make_decoder_copy(model.config.name_or_path)
    logits = decoder(hs)
    out : TraceResult = TraceResult(logits, list(range(len(model.transformer.h))), len(model.transformer.h))
    logits : LogitResult = out.decode_logits(prompt_idx=[0,1])
    
    tok_a_f = logits[-1,0,-1].tokens(model.tokenizer)
    tok_b_f = logits[-1,1,-1].tokens(model.tokenizer)
    assert tok_a_f == ')', f"{repr(tok_a_f)}"
    assert tok_b_f == '6', f"{repr(tok_b_f)}"
    
def test_collect_at_token_idx4():
    prompts = [
        '<fim_prefix>print("hello world"<fim_suffix>\n<fim_middle>',
        "<fim_prefix>a=6\nb=6\nc=<fim_suffix><fim_middle>",
    ]
    toks = ["<fim_middle>"]
    target_fn = partial(mask_target_tokens, tokens=toks)
    hs = collect_hidden_states(model, prompts, get_target_ids=target_fn).to("cpu")
    decoder = make_decoder_copy(model.config.name_or_path)
    logits = decoder(hs)
    out : TraceResult = TraceResult(logits, list(range(len(model.transformer.h))), len(model.transformer.h))
    logits : LogitResult = out.decode_logits(prompt_idx=[0,1])

    tok_a_f = logits[-1,0,-1].tokens(model.tokenizer)
    tok_b_f = logits[-1,1,-1].tokens(model.tokenizer)
    assert tok_a_f == ')', f"{repr(tok_a_f)}"
    assert tok_b_f == '6', f"{repr(tok_b_f)}"
    
def test_collect_at_token_idx5():
    prompts = [
        '<fim_prefix>print("hello world"<fim_suffix>\n<fim_middle>',
        "<fim_prefix>a=6\nb=6\nc=<fim_suffix><fim_middle>",
    ]
    toks = [0,-1]
    target_fn = partial(mask_target_tokens, tokens=toks)
    hs = collect_hidden_states(model, prompts, get_target_ids=target_fn).to("cpu")
    decoder = make_decoder_copy(model.config.name_or_path)
    logits = decoder(hs)
    out : TraceResult = TraceResult(logits, list(range(len(model.transformer.h))), len(model.transformer.h))
    logits : LogitResult = out.decode_logits(prompt_idx=[0,1], token_idx=[0,-1])
    
    tok_a_f = logits[-1,0,-1].tokens(model.tokenizer)
    tok_b_f = logits[-1,1,-1].tokens(model.tokenizer)
    assert tok_a_f == ')', f"{repr(tok_a_f)}"
    assert tok_b_f == '6', f"{repr(tok_b_f)}"
    
def test_interp_patch():
    prompts = [
        '<fim_prefix>print("hello world"<fim_suffix>\n<fim_middle>',
        "<fim_prefix>a=6\nb=6\nc=<fim_suffix><fim_middle>",
    ]
    toks = ["<fim_prefix>", "<fim_suffix>", "<fim_middle>"]
    target_fn = partial(mask_target_tokens, tokens=toks)
    hs = collect_hidden_states(model, prompts, get_target_ids=target_fn)
    patch = hs.expand(-1,len(prompts),-1,-1)
    out = insert_patch(model, prompts, patch, list(range(len(model.transformer.h))), toks, patch_mode="subst")
    out : LogitResult = out.decode_logits(prompt_idx=[0,1])
    tok_a_f = out[-1,0,-1].tokens(model.tokenizer)
    tok_b_f = out[-1,1,-1].tokens(model.tokenizer)
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
    
    original_prompt = '''from typing import List

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
    prompt = original_prompt
    for i in tqdm(range(max_out)):
        res = logit_lens(model, prompt, -1)
        logits = res.decode_logits()
        tok = logits[-1,-1].tokens(model.tokenizer)
        if tok == "<|endoftext|>":
            break
        generated.append(tok)
        prompt = prompt + tok

    assert prompt.strip() == gold.strip(), f"\n{prompt.strip()}\n\n\n{gold}"
    
    generated = []
    prompt = original_prompt
    patch = torch.zeros(24, 1, 1, model.config.n_embd)
    for i in tqdm(range(max_out)):
        # patch nothing, this checks precision errors in code
        res = insert_patch(model, prompt, patch, list(range(len(model.transformer.h))), [-1], collect_hidden_states=False)
        logits = res.decode_logits()
        tok = logits[-1,-1].tokens(model.tokenizer)
        if tok == "<|endoftext|>":
            break
        generated.append(tok)
        prompt = prompt + tok
    
    assert prompt.strip() == gold.strip(), f"\n{prompt.strip()}\n\n\n{gold}"

def last_assert_statement_index(prompt_input_ids: List[int], **kwargs) -> List[int]:
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
    
def test_patch_last_func_name():
    funcA = """
def f(x):
   if x > 50:
       return True
   else:
       return False

assert f(51) == """
    
    funcB = """
def f(x):
   if x > 50:
       return 7
   elif x > 10:
       return 5
   else:
       return 2

assert f(51) == """
    patch = collect_hidden_states(model, funcB, last_assert_statement_index)
    patch = torch.randn(24,1,3,2048)
    out = insert_patch(model, funcA, patch, list(range(10,24)), 
                       last_assert_statement_index, patch_mode="subst")
    logits = out.decode_logits(top_k=10)
    prediction = logits[-1,0].tokens(model.tokenizer)
    assert prediction == "7", f"predicted: {prediction}"

"""
CLI running code
"""
def repeat_test(func, n, **kwargs):
    for i in range(n):
        print(f"Running {func.__name__} {i+1}/{n}")
        func(**kwargs)
        
# repeating tests multiple times ensures no precision errors in code

repeat_test(test_logit_pipeline, args.num_reps)
test_masked_fill()
test_masked_get()
test_mask_target_idx()
test_mask_target_token()
repeat_test(test_patch, args.num_reps)
repeat_test(test_logit_generation_match, args.num_reps)
repeat_test(test_collect_at_token_idx, args.num_reps)
repeat_test(test_collect_at_token_idx2, args.num_reps)
repeat_test(test_collect_at_token_idx3, args.num_reps)
repeat_test(test_collect_at_token_idx4, args.num_reps)
repeat_test(test_collect_at_token_idx5, args.num_reps)
repeat_test(test_interp_patch, args.num_reps)
repeat_test(test_generation, 1)
repeat_test(test_patch_last_func_name, 1)

print("All tests passed!")
