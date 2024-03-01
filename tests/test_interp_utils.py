from codetrace.interp_utils import *
from codetrace.interp_vis import *
from nnsight import LanguageModel
from nnsight import util
from nnsight.tracing import Proxy

# re-run this
# NOTE: make sure padding is left side for list of prompts,
# to enable -1 indexing (most common)

prompts = [
    'print(f',
    'a=0\nb=1\nc=',
]

model = "/home/arjun/models/starcoderbase-1b"
model = LanguageModel(model, device_map="cuda")
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
    
def test_attn_collect():
    # TODO: test
    pass
    # prompts = [
    #     '<fim_prefix>print("hello world"<fim_suffix><fim_middle>',
    #     "<fim_prefix>a=6\nb=6\nc=<fim_suffix><fim_middle>",
    # ]
    # hs = collect_attention_output(model, prompts[0])
    # out = insert_attn_patch(model, prompts, hs, 14, patch_mode="subst")
    # out : Logits = out.decode_logits(prompt_idx=[0,1])
    # tok_a_f = out[-1][0][-1].tokens(model.tokenizer)[-1]
    # tok_b_f = out[-1][1][-1].tokens(model.tokenizer)[-1]
    # assert tok_a_f == ')', f"{repr(tok_a_f)}"
    # # assert tok_b_f == ')', f"{repr(tok_b_f)}"
    # NOTE: this one is hard to sanity
    
def test_insert_patch_with_generation():
    prompt_a = """def is_prime(n):"""
    prompt_b = """def fib(n):"""
    patch_a = collect_hidden_states_at_tokens(model, prompt_a, -1, list(range(24)))
    patch_b = collect_hidden_states_at_tokens(model, prompt_b, -1, list(range(24)))
    print(patch_a.shape)
    print(patch_b.shape)
    
    normal_out_a = insert_patch_with_generation(model, prompt_a, patch_a, list(range(24)), -1, 128)
    normal_out_b = insert_patch_with_generation(model, prompt_b, patch_b, list(range(24)), -1, 128)
    
    patched_out_b = insert_patch_with_generation(model, prompt_b, patch_a, list(range(24)), -1, 128)
    
    def _decode(out):
        out : LogitResult = out.decode_logits(prompt_idx=[0])
        generated = []
        for tok in range(out.size(-1)):
            generated.append(out[-1][0][tok].tokens(model.tokenizer))
        print(generated)
        
    generated_patched_b = _decode(patched_out_b)
    generated_normal_b = _decode(normal_out_b)
    generated_normal_a = _decode(normal_out_a)
    print("Generated patched b")
    print(generated_patched_b)
    print("Generated normal b")
    print(generated_normal_b)
    print("Generated normal a")
    print(generated_normal_a)
    
def test_insert_patch_with_generation_multi():
    raise NotImplementedError("Not implemented")
    prompt_a = """def is_prime(n):"""
    prompt_b = """def fib(n):"""
    patch_a = collect_hidden_states_at_tokens(model, prompt_a, -1, list(range(24)))
    patch_b = collect_hidden_states_at_tokens(model, prompt_b, -1, list(range(24)))
    print(patch_a.shape)
    print(patch_b.shape)
    
    normal_out_a = insert_patch_with_generation(model, prompt_a, patch_a, list(range(24)), -1, 128)
    normal_out_b = insert_patch_with_generation(model, prompt_b, patch_b, list(range(24)), -1, 128)
    
    patched_out_b = insert_patch_with_generation(model, prompt_b, patch_a, list(range(24)), -1, 128)
    
    def _decode(out):
        out : LogitResult = out.decode_logits(prompt_idx=[0])
        generated = []
        for tok in range(out.size(-1)):
            generated.append(out[-1][0][tok].tokens(model.tokenizer))
        print(generated)
        
    generated_patched_b = _decode(patched_out_b)
    generated_normal_b = _decode(normal_out_b)
    generated_normal_a = _decode(normal_out_a)
    print("Generated patched b")
    print(generated_patched_b)
    print("Generated normal b")
    print(generated_normal_b)
    print("Generated normal a")
    print(generated_normal_a)
    
def test_insert_patch_with_generation_str():
    prompt_a = """def is_prime(n):"""
    prompt_b = """def fib(n):"""
    patch_a = collect_hidden_states_at_tokens(model, prompt_a, "def", list(range(24)))
    patch_b = collect_hidden_states_at_tokens(model, prompt_b, "def", list(range(24)))
    
    # _,normal_out_a = insert_patch_with_generation(model, prompt_a, patch_a, list(range(24)), "def", 120)
    _,normal_out_b = insert_patch_with_generation(model, prompt_b, patch_b, list(range(24)), "def", 36)
    print([model.tokenizer.decode(x) for x in normal_out_b])
    # print([model.tokenizer.decode(x) for x in normal_out_a.flatten().tolist()])
    # print([model.tokenizer.decode(x) for x in normal_out_b.flatten().tolist()])
    # patched_out_b = insert_patch_with_generation(model, prompt_b, patch_a, list(range(24)), "def", 128)
    
    # def _decode(out):
    #     out : LogitResult = out.decode_logits(prompt_idx=[0])
    #     generated = []
    #     for tok in range(out.size(-1)):
    #         generated.append(out[-1][0][tok].tokens(model.tokenizer))
    #     print(generated)
        
    # generated_patched_b = _decode(patched_out_b)
    # generated_normal_b = _decode(normal_out_b)
    # generated_normal_a = _decode(normal_out_a)
    # print("Generated patched b")
    # print(generated_patched_b)
    # print("Generated normal b")
    # print(generated_normal_b)
    # print("Generated normal a")
    # print(generated_normal_a)
    
def repeat_test(func, n):
    for i in range(n):
        print(f"Running test {func.__name__} {i+1}/{n}")
        func()
        
if __name__ == "__main__":
    # repeat_test(test_logit_pipeline, 1)
    # repeat_test(test_patch, 1)
    # repeat_test(test_patch_vis, 1)
    # repeat_test(test_patch_vis_mult, 1)
    # repeat_test(test_logit_generation_match, 1)
    # repeat_test(test_collect_at_token_idx, 1)
    # repeat_test(test_collect_at_token_idx2, 1)
    # repeat_test(test_interp_patch, 10)
    # test_attn_collect()
    test_insert_patch_with_generation_str()
    # test_insert_patch_with_generation_multi()
    # test_insert_patch_with_generation()
    print("All tests passed!")