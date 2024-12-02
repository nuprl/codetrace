from codetrace.utils import (
    masked_fill,
    masked_get,
    masked_add,
    mask_target_tokens,
    mask_target_idx,
    topk_filtering
)
from transformers import AutoTokenizer
import torch
import pytest
import einops
from codetrace.scripts.plot_activations import _chain_activations

def test_chain_activations():
    activs = [torch.randn(40,30,5,100)]*3 # nsamples,nlayer, nprompt, hdim
    output = _chain_activations(activs, 5)
    assert list(output.shape) == [120,5,100]

@pytest.mark.parametrize("logits, top_k, do_log_probs, expected_indices", [
    (torch.tensor([[1.0, 2.0, 3.0]]), 2, False, torch.tensor([[2, 1]])),
    (torch.tensor([[1.0, 2.0, 3.0]]), 1, False, torch.tensor([[2]])),
    (torch.tensor([[0.1, 0.3, 0.2]]), 2, False, torch.tensor([[1, 2]])),
    (torch.tensor([[1.0, 2.0, 3.0]]), 2, True, torch.tensor([[2, 1]])),
    (torch.tensor([[0.5, 0.3, 0.2]]), 3, False, torch.tensor([[0, 1, 2]])),
])
def test_topk_filtering(logits, top_k, do_log_probs, expected_indices):
    # Call the function with parameters
    result = topk_filtering(logits, top_k, do_log_probs)
    
    # Assert the indices and values match the expected ones
    assert torch.equal(result.indices, expected_indices), f"Expected indices {expected_indices}, got {result.indices}"

@pytest.mark.parametrize("logits, top_k, do_log_probs, expected_indices", [
    (torch.tensor([[0.5, 0.3, 0.2]]), 3, False, torch.tensor([[0, 1, 2]])),  # All elements are included, since top_k = 3
    (torch.tensor([[1.0, 2.0, 3.0]]), 1, False, torch.tensor([[2]])),  # Only the highest value (index 2) should be kept
    (torch.tensor([[1.0, 2.0, 3.0]]), 2, True, torch.tensor([[2, 1]])),  # log_softmax, 2 largest values should be kept
])
def test_edge_cases(logits, top_k, do_log_probs, expected_indices):
    # Test the basic filtering logic, testing edge cases like all logits being 0, or negative logits
    result = topk_filtering(logits, top_k, do_log_probs)
    assert torch.equal(result.indices, expected_indices), f"Expected indices {expected_indices}, got {result.indices}"

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

    src = torch.Tensor([[[1,1,1,1],
                         [2,2,2,2],
                         [3,3,3,3]],
                         [[4,4,4,4],
                          [5,5,5,5],
                          [6,6,6,6]]])
    mask = torch.BoolTensor([[1,1,0],[1,1,0]])
    mask = einops.repeat(mask, "l t -> l t d", d=4)
    patch = torch.Tensor([[[10,11,12,13],
                         [20,21,22,23],
                         [30,31,32,33]],
                         [[40,41,42,43],
                          [50,51,52,53],
                          [60,61,62,63]]])
    res = masked_fill(src, mask, patch)
    expected = torch.Tensor([[[10,11,12,13],
                         [20,21,22,23],
                         [3,3,3,3]],
                         [[40,41,42,43],
                          [50,51,52,53],
                          [6,6,6,6]]])
    assert torch.equal(res, expected)

    src = torch.Tensor([[[1,1,1,1],
                         [2,2,2,2],
                         [3,3,3,3]],
                         [[4,4,4,4],
                          [5,5,5,5],
                          [6,6,6,6]]])
    mask = torch.BoolTensor([[1,1,1],[1,1,1]])
    mask = einops.repeat(mask, "l t -> l t d", d=4)
    patch = torch.Tensor([[[10,11,12,13],
                         [20,21,22,23],
                         [30,31,32,33]],
                         [[40,41,42,43],
                          [50,51,52,53],
                          [60,61,62,63]]])
    res = masked_fill(src, mask, patch)
    expected = patch
    assert torch.equal(res, expected)


def test_masked_add():
    src = torch.Tensor([[1,2,3],[4,5,6]])
    mask = torch.BoolTensor([[True, False, True],[False,True,False]])
    patch = torch.Tensor([[10,20,30],[40,50,60]])
    res = masked_add(src, mask, patch)
    expected = torch.Tensor(
        [[11,2,33],
         [4,55,6]]
    )
    assert torch.equal(res, expected), f"{res}!={expected}"

    src = torch.Tensor([[1,2,3],[4,5,6]])
    mask = torch.BoolTensor([[1,1,1],[1,1,1]])
    patch = torch.Tensor([[10,20,30],[40,50,60]])
    res = masked_add(src, mask, patch)
    expected = torch.Tensor([[11,22,33],[44,55,66]])
    assert torch.equal(res, expected), f"{res}!={expected}"

    src = torch.Tensor([[[1,1,1,1],
                         [2,2,2,2],
                         [3,3,3,3]],
                         [[4,4,4,4],
                          [5,5,5,5],
                          [6,6,6,6]]])
    mask = torch.BoolTensor([[1,1,0],[1,1,0]])
    mask = einops.repeat(mask, "l t -> l t d", d=4)
    patch = torch.Tensor([[[10,11,12,13],
                         [20,21,22,23],
                         [30,31,32,33]],
                         [[40,41,42,43],
                          [50,51,52,53],
                          [60,61,62,63]]])
    res = masked_add(src, mask, patch)
    expected = torch.Tensor([[[11,12,13,14],
                         [22,23,24,25],
                         [3,3,3,3]],
                         [[44,45,46,47],
                          [55,56,57,58],
                          [6,6,6,6]]])
    assert torch.equal(res, expected), f"{res}!={expected}"

    src = torch.Tensor([[[1,1,1,1],
                         [2,2,2,2],
                         [3,3,3,3]],
                         [[4,4,4,4],
                          [5,5,5,5],
                          [6,6,6,6]]])
    mask = torch.BoolTensor([[1,1,1],[1,1,1]])
    mask = einops.repeat(mask, "l t -> l t d", d=4)
    expected = torch.Tensor([[[11,12,13,14],
                         [22,23,24,25],
                         [33,34,35,36]],
                         [[44,45,46,47],
                          [55,56,57,58],
                          [66,67,68,69]]])
    res = masked_add(src, mask, patch)
    assert torch.equal(res, expected), f"{res}!={expected}"

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
    tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoderbase-1b")
    tokens = ["<fim_middle>","<fim_prefix>"]
    prompts = [
        "<fim_prefix><fim_middle>x",
        "d<fim_middle>e"
    ]
    input_ids = tokenizer(prompts, return_tensors="pt")["input_ids"]
    assert input_ids.shape == (2, 3)
    token_ids = tokenizer(tokens, return_tensors="pt")["input_ids"]
    result = mask_target_tokens(input_ids, token_ids)
    expected = torch.BoolTensor([
        [True, True, False],
        [False, True, False]
    ])
    assert torch.equal(result, expected), f"result {result}, expected {expected}"

def test_mask_target_idx():
    tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoderbase-1b")
    indices = [0,2]
    prompts = [
        "a b x",
        "d c e"
    ]
    input_ids = tokenizer(prompts, return_tensors="pt")["input_ids"]
    assert input_ids.shape == (2, 3)
    result = mask_target_idx(input_ids, indices)
    expected = torch.BoolTensor([
        [True, False, True],
        [True, False, True],
    ])
    assert torch.equal(result, expected), f"result {result}, expected {expected}"

if __name__ == "__main__":
    import pytest
    import os
    pytest.main([os.path.abspath(__file__), "-vv"])