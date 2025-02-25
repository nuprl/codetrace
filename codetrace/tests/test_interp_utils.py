from argparse import ArgumentParser
from functools import partial
from nnsight import LanguageModel
import torch
from codetrace.interp_utils import (
    collect_hidden_states, 
    logit_lens,
    insert_patch,
    LogitResult,
    TraceResult,
    _prepare_layer_patch
)
from codetrace.utils import mask_target_idx, mask_target_tokens, predict, copy_decoder
from codetrace.batched_utils import batched_get_averages, batched_insert_patch_logit
"""
Setup code
"""
parser = ArgumentParser()
parser.add_argument("--modelname", type=str, default="bigcode/starcoderbase-1b")
parser.add_argument("--num_reps", type=int, default=10)
args = parser.parse_args()
model = LanguageModel(args.modelname, device_map="cuda", torch_dtype=torch.bfloat16, dispatch=True)

"""
tests
"""
def test_prepare_layer_patch():
    tensor = torch.Tensor([ # 2 prompts, 3 toks, ndim 1
        [[1.],[4.],[8.]],
        [[7.],[2.],[6.]]
    ])
    patch = torch.Tensor([ # 2 prompts, 1 toks, ndim 1
        [[9.]],
        [[-1.]]
    ])
    expected = torch.Tensor([ # 2 prompts, 3 toks, ndim 1
        [[0.],[0.],[9.]],
        [[0.],[0.],[-1.]]
    ])
    output = _prepare_layer_patch(patch, n_tokens=tensor.shape[1])
    assert torch.equal(output, expected),output

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

def patch_clean_to_corrupt(model, clean, corrupt, indices_or_tokens):
    if isinstance(indices_or_tokens[0], str):
        indices = model.tokenizer(indices_or_tokens, return_tensors="pt")["input_ids"]
        targets = partial(mask_target_tokens, token_ids=indices)
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
    logits : LogitResult = logits.decode_logits(prompt_idx=[0,1], token_idx=[-1])
    tok_a_f = logits[0,0,0].tokens(model.tokenizer)
    tok_b_f = logits[0,1,0].tokens(model.tokenizer)

    toks = predict(model, model.tokenizer,prompts)
    assert toks[0] == tok_a_f, f"{toks[0]} != {tok_a_f}"
    assert toks[1] == tok_b_f, f"{toks[1]} != {tok_b_f}"
    
    
def test_collect_at_token_idx():
    prompts = [
        '<fim_prefix>print("hi"<fim_suffix>\n<fim_middle>',
        "<fim_prefix>a=6\nb=6\nc=<fim_suffix><fim_middle>",
    ]
    logits = logit_lens(model, prompts)
    logits : LogitResult = logits.decode_logits(prompt_idx=[0,1], token_idx=[-1])

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
    token_ids = model.tokenizer(toks, return_tensors="pt")["input_ids"]
    target_fn = partial(mask_target_tokens, token_ids=token_ids)
    hs = collect_hidden_states(model, prompts, target_fn=target_fn).to("cpu")
    decoder = copy_decoder(model.config.name_or_path, torch.bfloat16)
    logits = decoder(hs)
    out : TraceResult = TraceResult(logits, list(range(len(model.transformer.h))), len(model.transformer.h))
    logits : LogitResult = out.decode_logits(prompt_idx=[0,1], token_idx=[-1])

    tok_a_f = logits[-1,0,-1].tokens(model.tokenizer)
    tok_b_f = logits[-1,1,-1].tokens(model.tokenizer)
    assert tok_a_f == ')', f"{repr(tok_a_f)}"
    assert tok_b_f == '6', f"{repr(tok_b_f)}"
    
    
def test_collect_at_token_idx3():
    prompts = [
        '<fim_prefix>print("hello world"<fim_suffix>\n<fim_middle>',
        "<fim_prefix>a=6\nb=6\nc=<fim_suffix><fim_middle>",
    ]
    target_fn = partial(mask_target_idx, indices=[-1])
    hs = collect_hidden_states(model, prompts, target_fn=target_fn).to("cpu")
    decoder = copy_decoder(model.config.name_or_path, torch.bfloat16)
    logits = decoder(hs)
    out : TraceResult = TraceResult(logits, list(range(len(model.transformer.h))), len(model.transformer.h))
    logits : LogitResult = out.decode_logits(prompt_idx=[0,1], token_idx=[-1])
    
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
    indices = model.tokenizer(toks, return_tensors="pt")["input_ids"]
    target_fn = partial(mask_target_tokens, token_ids=indices)
    hs = collect_hidden_states(model, prompts, target_fn=target_fn).to("cpu")
    decoder = copy_decoder(model.config.name_or_path, torch.bfloat16)
    logits = decoder(hs)
    out : TraceResult = TraceResult(logits, list(range(len(model.transformer.h))), len(model.transformer.h))
    logits : LogitResult = out.decode_logits(prompt_idx=[0,1], token_idx=[-1])

    tok_a_f = logits[-1,0,-1].tokens(model.tokenizer)
    tok_b_f = logits[-1,1,-1].tokens(model.tokenizer)
    assert tok_a_f == ')', f"{repr(tok_a_f)}"
    assert tok_b_f == '6', f"{repr(tok_b_f)}"
    
def test_collect_at_token_idx5():
    prompts = [
        '<fim_prefix>print("hello world"<fim_suffix>\n<fim_middle>',
        "<fim_prefix>a=6\nb=6\nc=<fim_suffix><fim_middle>",
    ]
    target_fn = partial(mask_target_idx, indices=[0,-1])
    hs = collect_hidden_states(model, prompts, target_fn=target_fn).to("cpu")
    decoder = copy_decoder(model.config.name_or_path, torch.bfloat16)
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
    indices = model.tokenizer(toks, return_tensors="pt")["input_ids"]
    target_fn = partial(mask_target_tokens, token_ids=indices)
    hs = collect_hidden_states(model, prompts, target_fn=target_fn)
    patch = hs.expand(-1,len(prompts),-1,-1)
    out = insert_patch(
        model, 
        prompts[::-1], 
        patch, 
        list(range(len(model.transformer.h))), 
        target_fn=target_fn)
    out : LogitResult = out.decode_logits(prompt_idx=[0,1], token_idx=[-1])
    tok_b_f = out[:,0].tokens(model.tokenizer)
    tok_a_f = out[:,1].tokens(model.tokenizer)
    assert tok_a_f == '6', f"{repr(tok_a_f)}"
    assert tok_b_f == ')', f"{repr(tok_b_f)}"
   
"""
CLI running code
"""
def repeat_test(func, n, **kwargs):
    for i in range(n):
        print(f"Running {func.__name__} {i+1}/{n}")
        func(**kwargs)
        
if __name__ == "__main__":
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
    test_prepare_layer_patch()
    print("All tests passed!")
