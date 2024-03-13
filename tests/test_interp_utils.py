from tqdm import tqdm
from codetrace.interp_utils import *
from codetrace.interp_vis import *
from codetrace.utils import *
from nnsight import LanguageModel
from nnsight import util
from nnsight.tracing import Proxy
from transformers import AutoModelForCausalLM, AutoTokenizer
import regex as re

# re-run this
# NOTE: make sure padding is left side for list of prompts,
# to enable -1 indexing (most common)

prompts = [
    'print(f',
    'a=0\nb=1\nc=',
]

modelname = "/home/arjun/models/starcoderbase-1b"
model = LanguageModel(modelname, device_map="cuda")
model.tokenizer.padding_side = "left"

def test_logit_pipeline():
    hs = collect_hidden_states(model, prompts)

    assert hs.shape[0] == 24, hs.shape # layers
    assert hs.shape[1] == 2, hs.shape # prompt len
    # 2 is token count, padded
    assert hs.shape[3] == model.config.n_embd, hs.shape
    
    logits = logit_lens(model, prompts)
    logits : LogitResult = logits.decode_logits( layers=[0,23], prompt_idx=[0,1])
    tok_a_f = logits[1][0].tokens(model.tokenizer)
    tok_b_f = logits[1][-1].tokens(model.tokenizer)
    assert tok_b_f == ['2'], tok_b_f
    assert tok_a_f == ['"'], tok_a_f
    
def test_patch():
    trace_res = patch_clean_to_corrupt(model, prompts[0], prompts[1], list(range(24)))
    tok_pred = trace_res.decode_logits().tokens(model.tokenizer)
    assert tok_pred == ['"'], tok_pred
    trace_res = patch_clean_to_corrupt(model, prompts[1], prompts[0], list(range(24)))
    tok_pred = trace_res.decode_logits().tokens(model.tokenizer)
    assert tok_pred == ['2'], tok_pred
    trace_res = patch_clean_to_corrupt(model, prompts[0], prompts[1], 0)
    tok_pred = trace_res.decode_logits().tokens(model.tokenizer)
    assert tok_pred != ['2'] and tok_pred != ['"'], tok_pred
    trace_res = patch_clean_to_corrupt(model, prompts[0], prompts[1], [])
    tok_pred = trace_res.decode_logits().tokens(model.tokenizer)
    assert tok_pred == ['2'], tok_pred
    
    
def test_patch_vis():
    patch_l = [1, 14,23]
    trace_results = []
    for l in patch_l:
        trace_res = patch_clean_to_corrupt(model, prompts[0], prompts[1], l)
        trace_results.append(trace_res)
    patched_heatmap_prediction(model, prompts[0], prompts[1], trace_results, patch_l)
    
    
def test_patch_vis_mult():
    cleans = [prompts[0], prompts[1]]
    corrs = [prompts[1], prompts[0]]
    patch_l = [1, 14,23]
    trace_results = []
    for l in patch_l:
        trace_res = patch_clean_to_corrupt(model, cleans, corrs, l)
        trace_results.append(trace_res)
    
    patched_heatmap_prediction(model, cleans,corrs, trace_results, patch_l, figtitle="test_fig")

def test_logit_generation_match():
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
    toks = ["<fim_prefix>", "<fim_suffix>", "<fim_middle>"]
    tok_idx = [model.tokenizer.encode(t)[0] for t in toks]
    logits = logit_lens(model, prompts)
    logits : LogitResult = logits.decode_logits(prompt_idx=[0,1])
    
    tok_a_f = logits[-1][0][-1].tokens(model.tokenizer)[-1]
    tok_b_f = logits[-1][1][-1].tokens(model.tokenizer)[-1]
    assert tok_a_f == ')', f"{repr(tok_a_f)}"
    assert tok_b_f == '6', f"{repr(tok_b_f)}"
    
def test_collect_at_token_idx2():
    prompts = [
        '<fim_prefix>print("hello world"<fim_suffix>\n<fim_middle>',
        "<fim_prefix>a=6\nb=6\nc=<fim_suffix><fim_middle>",
    ]
    toks = ["<fim_prefix>", "<fim_suffix>", "<fim_middle>"]
    tok_idx = [model.tokenizer.encode(t)[0] for t in toks]
    out = collect_hidden_states_at_tokens(model, prompts, toks[-1], debug=True)
    out : TraceResult = TraceResult(out, list(range(len(model.transformer.h))))
    logits : LogitResult = out.decode_logits(prompt_idx=[0,1])
    
    tok_a_f = logits[-1][0][-1].tokens(model.tokenizer)[-1]
    tok_b_f = logits[-1][1][-1].tokens(model.tokenizer)[-1]
    assert tok_a_f == ')', f"{repr(tok_a_f)}"
    assert tok_b_f == '6', f"{repr(tok_b_f)}"
    
def test_interp_patch():
    prompts = [
        '<fim_prefix>print("hello world"<fim_suffix>\n<fim_middle>',
        "<fim_prefix>a=6\nb=6\nc=<fim_suffix><fim_middle>",
    ]
    toks = ["<fim_prefix>", "<fim_suffix>", "<fim_middle>"]
    hs = collect_hidden_states_at_tokens(model, prompts[0], toks)
    out = insert_patch(model, prompts, hs, list(range(len(model.transformer.h))), toks, patch_mode="subst")
    out : LogitResult = out.decode_logits(prompt_idx=[0,1])
    tok_a_f = out[-1][0][-1].tokens(model.tokenizer)[-1]
    tok_b_f = out[-1][1][-1].tokens(model.tokenizer)[-1]
    assert tok_a_f == ')', f"{repr(tok_a_f)}"
    assert tok_b_f == ')', f"{repr(tok_b_f)}"
   
    
   
def repeat_test(func, n):
    for i in range(n):
        print(f"Running test {func.__name__} {i+1}/{n}")
        func()

def test_topk_topp():
    ex = [
        [[0.9, 0.05, 0.04, 0.01]],
        [[9, 0.1, 0.02, 0.001]],
        [[0.5, 0.25, 0.25, 0.1]]
    ]
    ## cumulative after softmax
    # tensor([[[[0.4422, 0.6313, 0.8184, 1.0000]],

    #      [[0.2612, 0.5223, 0.7634, 1.0000]],

    #      [[0.3098, 0.5511, 0.7923, 1.0000]]]])
    ## cumulative with k select first
    # tensor([[[[0.5404, 0.7713, 1.0000]],

    #      [[0.9997, 0.9999, 1.0000]],

    #      [[0.3910, 0.6955, 1.0000]]]])
    top_p = 0.8
    top_k = 3
    input_t = torch.tensor(ex)
    inputs = torch.stack([input_t], dim=0)
    assert inputs.shape == (1, 3, 1, 4), inputs.shape
    logits = top_k_top_p_filtering(inputs, top_k, top_p, do_log_probs=False)
    # logits_log = top_k_top_p_filtering(inputs, top_k, top_p, do_log_probs=True)
    assert list(logits.values[0,0,0]) == [0.9,0.05], logits.values[0,0,0]
    assert list(logits.values[0,1,0]) == [9,9], logits.values[0,1,0]
    assert list(logits.values[0,2,0]) == [0.5, 0.25], logits.values[0,2,0]
    assert list(logits.indices[0,0,0]) == [0,1], logits.indices[0,0,0]
    assert list(logits.indices[0,1,0]) == [0,0], logits.indices[0,1,0]
    assert list(logits.indices[0,2,0]) == [0,1], logits.indices[0,2,0]
    
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
    
    prompt = ['''from typing import List

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
                max_depth = max''']
    
    max_out = 512
    generated = []
    for i in tqdm(range(max_out)):
        res = logit_lens(model, prompt, -1)
        logits = res.decode_logits()
        tok = logits[-1][-1].tokens(model.tokenizer)[0]
        if tok == "<|endoftext|>":
            break
        generated.append(tok)
        prompt = [prompt[0] + tok]

    assert prompt[0].strip() == gold.strip(), f"\n{prompt[0]}\n{gold}\n{generated}"
    
    generated = []
    patch = torch.zeros(24, 1, 1, model.config.n_embd)
    for i in tqdm(range(max_out)):
        res = insert_patch(model, prompt, patch, [], [-1], collect_hidden_states=False)
        logits = res.decode_logits()
        tok = logits[-1][-1].tokens(model.tokenizer)[0]
        if tok == "<|endoftext|>":
            break
        generated.append(tok)
        prompt = [prompt[0] + tok]
    
    assert prompt[0].strip() == gold.strip(), f"\n{prompt[0]}\n{gold}"
    
            
            
if __name__ == "__main__":
    repeat_test(test_logit_pipeline, 1)
    repeat_test(test_patch, 1)
    repeat_test(test_patch_vis, 1)
    repeat_test(test_patch_vis_mult, 1)
    repeat_test(test_logit_generation_match, 1)
    repeat_test(test_collect_at_token_idx, 1)
    repeat_test(test_collect_at_token_idx2, 1)
    repeat_test(test_interp_patch, 10)
    # test_topk_topp()
    test_generation()
    print("All tests passed!")
    
    
    