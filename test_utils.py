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
    logits : Logit = logits.decode_logits(model, layers=[0,23], prompt_idx=[0,1])
    tok_a_f = logits[1][0].tokens()
    tok_b_f = logits[1][-1].tokens()
    assert tok_b_f == ['2'], tok_b_f
    assert tok_a_f == ['"'], tok_a_f
    
def test_patch():
    trace_res = patch_clean_to_corrupt(model, prompts[0], prompts[1], -1, -1, list(range(24)))
    tok_pred = trace_res.score_top(1).get_tokens(model, layers = [-1], prompt_idx=0)[-1]
    assert tok_pred == "world", tok_pred
    trace_res = patch_clean_to_corrupt(model, prompts[1], prompts[0], -1, -1, list(range(24)))
    tok_pred = trace_res.score_top(1).get_tokens(model, layers = [-1], prompt_idx=0)[-1]
    assert tok_pred == "0", tok_pred
    trace_res = patch_clean_to_corrupt(model, prompts[0], prompts[1], -1, -1, [2])
    tok_pred = trace_res.score_top(1).get_tokens(model, layers = [-1], prompt_idx=0)[-1]
    assert tok_pred != "world", tok_pred
    
def test_patch_vis():
    trace_res = patch_clean_to_corrupt(model, prompts[0], prompts[1], -1, -1, [14,23])
    patch_logit_heatmap(model, prompts[0], prompts[1], trace_res, [1, 3, 7, 14,23], -1,-1)
    
if __name__ == "__main__":
    test_logit_pipeline()
    # test_patch()
    # test_patch_vis()
    print("All tests passed!")