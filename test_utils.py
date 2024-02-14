from interp_utils import *
from nnsight import LanguageModel

# NOTE: make sure padding is left side for list of prompts,
# to enable -1 indexing (most common)

prompts = [
    'print("hello_',
    'n = 0\nassert(n == ',
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
    tok_world = logits.score_top(1).get_tokens(model, layers = list(range(24)), prompt_idx=0)[-1]
    tok_0 = logits.score_top(1).get_tokens(model, layers = list(range(24)), prompt_idx=1)[-1]
    assert tok_world == "world", tok_world
    assert tok_0 == "0", tok_0
    
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
    
if __name__ == "__main__":
    test_logit_pipeline()
    test_patch()
    print("All tests passed!")