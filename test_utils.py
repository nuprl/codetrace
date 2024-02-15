from interp_utils import *
from interp_vis import *
from nnsight import LanguageModel

# re-run this
# NOTE: make sure padding is left side for list of prompts,
# to enable -1 indexing (most common)

prompts = [
    'print(f',
    'a=0\nb=1\nc=',
]


model = "/home/arjun/models/starcoderbase-1b"
model = LanguageModel(model, device_map="cuda:0")
model.tokenizer.padding_side = "left"

def test_logit_pipeline():
    hs = collect_hidden_states(model, prompts)

    assert hs.shape[0] == 24, hs.shape # layers
    assert hs.shape[1] == 2, hs.shape # prompt len
    # 2 is token count, padded
    assert hs.shape[3] == model.config.n_embd, hs.shape
    
    logits = logit_lens(model, prompts)
    logits : Logit = logits.decode_logits( layers=[0,23], prompt_idx=[0,1])
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
    logits : Logit = logits.decode_logits(prompt_idx=[0,1])
    tok_a_f = logits[0][0].tokens(model.tokenizer)[0]
    tok_b_f = logits[0][1].tokens(model.tokenizer)[0]
    
    with model.generate(max_new_tokens=1) as gen:
        with gen.invoke(prompts) as invoker:
            invoker.next()

    output = gen.output
    toks = [model.tokenizer.decode(x) for x in output[:,-1]]
    assert toks[0] == tok_a_f, f"{toks[0]} != {tok_a_f}"
    assert toks[1] == tok_b_f, f"{toks[1]} != {tok_b_f}"
    
if __name__ == "__main__":
    test_logit_pipeline()
    test_patch()
    test_patch_vis()
    test_patch_vis_mult()
    test_logit_generation_match()
    print("All tests passed!")