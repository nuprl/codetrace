from codetrace.utils import (
    masked_fill,
    masked_get,
    masked_add,
    mask_target_tokens,
    mask_target_idx,
    top_k_top_p_filtering
)
from transformers import AutoTokenizer
import torch

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
