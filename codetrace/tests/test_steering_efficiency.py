import uuid
from codetrace.interp_utils import collect_hidden_states, TraceResult, _prepare_layer_patch
from codetrace.batched_utils import batched_get_averages, batched_insert_patch_logit, _resize_patch
from codetrace.utils import reset_index_dim0, predict, copy_decoder, load_dataset
from codetrace.utils import masked_add, mask_target_idx, masked_fill, mask_target_tokens, masked_get
from codetrace.steering import SteeringManager, subtract_avg
from codetrace.scripts.launch_steer import main as launch_steer
import torch
from nnsight import LanguageModel
from functools import partial

MODEL = "/mnt/ssd/arjun/models/starcoderbase-1b"
model = LanguageModel(MODEL, device_map="cuda", torch_dtype=torch.bfloat16)
tokenizer = model.tokenizer

def test_masked_fill():
    output = masked_fill(torch.Tensor([1,2,3]), torch.BoolTensor([1,0,0]), torch.Tensor([4,5,6]))
    expected = torch.Tensor([4,2,3])
    assert torch.equal(output, expected)

def test_masked_add():
    output = masked_add(torch.Tensor([1,2,3]), torch.BoolTensor([1,0,0]), torch.Tensor([4,6,7]))
    expected = torch.Tensor([5,2,3])
    assert torch.equal(output, expected)

def test_masked_get():
    output = masked_get(torch.Tensor([1,2,3]), torch.BoolTensor([0,1,0]))
    expected = torch.Tensor([0,2,0])
    assert torch.equal(output, expected)

def test_mask_target_token():
    output = mask_target_tokens(torch.Tensor([1,2,3]),[3,1])
    expected = torch.Tensor([1,0,1])
    assert torch.equal(output, expected)

def test_mask_target_idx():
    output = mask_target_idx(torch.Tensor([1,2,3]),[2,1],0)
    expected = torch.Tensor([0,1,1])
    assert torch.equal(output, expected)

def test_reset_index_dim0():
    output = reset_index_dim0(torch.Tensor([0.1,0.2,0.3]), [3,7,1], 9)
    expected = torch.Tensor([0., 0.3, 0., 0.1, 0., 0., 0., 0.2, 0.])
    assert torch.equal(output, expected)

    output = reset_index_dim0(torch.Tensor([[0.1,0.2,0.3],[4., 9., 1.],[5.,6.,2.]]), [0,4,3], 5)
    expected = torch.Tensor([[0.1,0.2,0.3],[0.,0.,0.],[0.,0.,0.],[5.,6.,2.],[4., 9., 1.]])
    assert torch.equal(output, expected)

def _check_reset_index(output, expected_shape, n_layers, indices, activations):
    assert list(output.shape) == expected_shape
    for i in range(n_layers):
        if i in indices:
            assert torch.equal(output[i],activations[indices.index(i)])
            assert output[i].sum().item() != 0
        else:
            assert output[i].sum().item() == 0

def test_reset_index():
    activations = torch.rand(3, 4, 2, 500)
    indices = [10,11,12]
    n_layers = 24
    output = reset_index_dim0(activations, indices, n_layers)
    expected_shape = [n_layers,4,2,500]
    _check_reset_index(output, expected_shape, n_layers, indices, activations)
    
    activations = torch.rand(5, 6, 3000)
    indices = [0, 1, 20, 11, 12]
    n_layers = 30
    output = reset_index_dim0(activations, indices, n_layers)
    expected_shape = [n_layers,6,3000]
    _check_reset_index(output, expected_shape, n_layers, indices, activations)

    activations = torch.rand(24, 4, 2, 500)
    indices = list(range(24))
    n_layers = 24
    output = reset_index_dim0(activations, indices, n_layers)
    assert torch.equal(output, activations)

def test_collect_hidden_states():
    layers = [5,16,23]
    prompts = [
        "y A","y B"
    ]
    output_all_layers = collect_hidden_states(
        model,
        prompts
    )
    output_some_layers = collect_hidden_states(
        model,
        prompts,
        layers
    )
    for l in range(output_all_layers.shape[0]):
        if l in layers:
            assert torch.equal(output_some_layers[l], output_all_layers[l])
        else:
            assert output_some_layers[l].sum().item() == 0


def test_collect_hidden_states_reduction():
    prompts = [
        "y A","y B"
    ]
    output_reduced = collect_hidden_states(
        model,
        prompts,
        target_fn=partial(mask_target_idx, indices=[-1]),
        reduction="sum"
    )
    output = collect_hidden_states(
        model,
        prompts,
        target_fn=partial(mask_target_idx, indices=[-1]),
    )
    output_reduced_max = collect_hidden_states(
        model,
        prompts,
        target_fn=partial(mask_target_idx, indices=[-1]),
        reduction="max"
    )
    assert torch.allclose(output_reduced, output[:,:,-1,:].squeeze(), atol=1e-4, rtol=1e-5)
    assert not torch.allclose(output_reduced_max, output[:,:,-1,:].squeeze(), atol=1e-4, rtol=1e-5)


def test_collect_hidden_states_target_fn():
    prompts = [
        "yabba yabba A","yabba yabba A"
    ]
    output_reduced = collect_hidden_states(
        model,
        prompts,
        target_fn=partial(mask_target_idx, indices=[-1]),
        reduction="sum"
    )
    output = collect_hidden_states(
        model,
        prompts
    )
    assert torch.allclose(output_reduced, output[:,:,-1,:].squeeze())


def test_batched_get_average():
    prompts = [
        "tr B","tr B"
    ]
    activations = collect_hidden_states(
        model,
        prompts,
        target_fn=partial(mask_target_idx, indices=[-1]),
        layers=[15]
    ).to("cpu")
    output_normal_avg = batched_get_averages(
        model,
        prompts,
        target_fn=partial(mask_target_idx, indices=[-1]),
        batch_size=2,
        layers=[15],
        reduction="sum"
    )
    output_sub_avg = batched_get_averages(
        model,
        prompts,
        target_fn=partial(mask_target_idx, indices=[-1]),
        batch_size=2,
        average_fn=subtract_avg,
        layers=[15],
        reduction="sum"
    )
    assert activations.sum().item() != 0
    assert output_normal_avg.sum().item() != 0
    assert output_sub_avg.sum().item() == 0

    mean_activ = activations.mean(dim=1)[:,1].squeeze()
    assert torch.allclose(mean_activ, output_normal_avg.squeeze(), atol=1e-4, rtol=1e-5)
    assert torch.allclose(output_normal_avg[15], activations[15,0,-1], atol=1e-4, rtol=1e-5)

    # 2(b-a)/2 == b-a
    prompts = [
        "whoop B","whoop A","whoop B","whoop A"
    ]
    output_sub_avg = batched_get_averages(
        model,
        prompts,
        target_fn=partial(mask_target_idx, indices=[-1]),
        batch_size=2,
        average_fn=subtract_avg,
        layers=[15],
        reduction="sum"
    )
    activ_A_B = collect_hidden_states(
        model,
        prompts[:2],
        target_fn=partial(mask_target_idx, indices=[-1]),
        layers=[15]
    ).to("cpu")
    assert output_sub_avg.sum().item() != 0 
    assert activ_A_B.sum().item() != 0 
    activ_A = activ_A_B[:,1,-1]
    activ_B = activ_A_B[:,0,-1]
    
    activ_BminA = activ_B - activ_A
    assert torch.allclose(activ_BminA, output_sub_avg.squeeze(), atol=1e-4, rtol=1e-5)


def test_insert_patched_logits():
    promptsA = [
        "def subtract(a:int, b: ", "def substr(s:str, sub: "
    ]
    promptsB = [
        "def print_name(prefix:str, name: ", "def add(a:int, b: "
    ]
    patchA = collect_hidden_states(
        model,
        promptsA,
        target_fn=partial(mask_target_idx, indices=[-1]),
        reduction="sum"
    ).unsqueeze(2)
    patchB = collect_hidden_states(
        model,
        promptsB,
        target_fn=partial(mask_target_idx, indices=[-1]),
        reduction="sum"
    ).unsqueeze(2)
    decoder = copy_decoder(MODEL, dtype=torch.bfloat16).to("cuda")
    logitsA = decoder(patchA)
    logitsB = decoder(patchB)
    print(patchA.shape, patchB.shape)
    resA = TraceResult(logitsA, list(range(patchA.shape[0])), patchA.shape[0])
    resB = TraceResult(logitsB, list(range(patchB.shape[0])), patchB.shape[0])
    
    print(resB.decode_logits([0],[-2]).tokens(tokenizer))
    predA = predict(model, tokenizer, promptsA)
    predB = predict(model, tokenizer, promptsB)
    print(predA, predB,)
    
    assert resA.decode_logits([0],[-2]).tokens(tokenizer).strip() == "int"
    assert resA.decode_logits([1],[-2]).tokens(tokenizer).strip() == "str"
    assert resB.decode_logits([1],[-2]).tokens(tokenizer).strip() == "int"
    assert resB.decode_logits([0],[-2]).tokens(tokenizer).strip() == "str"

    patched_prediction = batched_insert_patch_logit(
        model,
        promptsB,
        patchA,
        target_fn=partial(mask_target_idx, indices=[-1]),
        patch_fn=masked_fill,
        layers_to_patch=list(range(0,24)),
        batch_size=2
    )
    assert [p.strip() for p in patched_prediction] == ["int","str"]
    patched_prediction = batched_insert_patch_logit(
        model,
        promptsA,
        patchB,
        target_fn=partial(mask_target_idx, indices=[-1]),
        patch_fn=masked_fill,
        layers_to_patch=list(range(10,20)),
        batch_size=2
    )
    assert [p.strip() for p in patched_prediction] == ["str","int"]


def test_prepare_patch():
    patch_reduced = torch.tensor(
        [
            [[3.],[4.]], #layer1
            [[5.],[6.]], #layer2
            [[-1.],[-4]], #layer 3
        ]
    )
    expected = torch.tensor(
        [
            [[[0.],[0.],[0.],[0.],[0.],[3.],[4.]],  [[0.],[0.],[0.],[0.],[0.],[3.],[4.]],    [[0.],[0.],[0.],[0.],[0.],[3.],[4.]],    [[0.],[0.],[0.],[0.],[0.],[3.],[4.]]], #layer1
            [[[0.],[0.],[0.],[0.],[0.],[5.],[6.]],  [[0.],[0.],[0.],[0.],[0.],[5.],[6.]],    [[0.],[0.],[0.],[0.],[0.],[5.],[6.]],    [[0.],[0.],[0.],[0.],[0.],[5.],[6.]]], #layer2
            [[[0.],[0.],[0.],[0.],[0.],[-1.],[-4]], [[0.],[0.],[0.],[0.],[0.],[-1.],[-4]],   [[0.],[0.],[0.],[0.],[0.],[-1.],[-4]],   [[0.],[0.],[0.],[0.],[0.],[-1.],[-4]]] #layer 3
        ]
    )
    assert list(patch_reduced.shape) == [3,2,1] # l, t, d
    resized = _resize_patch(patch_reduced, 4)
    assert list(resized.shape) == [3,4,2,1] # l,p,t,d
    output = _prepare_layer_patch(resized[-1],7)
    assert list(output.shape) == [4,7,1]
    assert torch.equal(output, expected[-1])


def _test_launch_steer():
    candidates = "nuprl-staging/type-steering"
    subset = "mutations-py-types_delete-starcoderbase-1b"
    output_dir = f"/tmp/codetrace_{uuid.uuid4()}_test_steer"
    launch_steer(
        MODEL,
        "bfloat16",
        candidates,
        [10,11,12],
        output_dir,
        "steer_split",
        "test_split",
        "steering_tensor.pt",
        2,
        2,
        100,
        10,
        subset,
        "train",
        ["test"]
    )
    output_dir2 = f"/tmp/codetrace_{uuid.uuid4()}_test_steer"
    launch_steer(
        MODEL,
        "bfloat16",
        candidates,
        [10,11,12],
        output_dir2,
        "steer_split",
        "test_split",
        "steering_tensor.pt",
        2,
        2,
        100,
        10,
        subset,
        "train",
        ["test"]
    )
    steering_tensor1 = torch.load(f"{output_dir}/steering_tensor.pt")
    steering_tensor2 = torch.load(f"{output_dir2}/steering_tensor.pt")
    test_split1 = load_dataset(f"{output_dir}/test_split").to_pandas()
    test_split2 = load_dataset(f"{output_dir2}/test_split").to_pandas()
    assert torch.equal(steering_tensor1, steering_tensor2)
    assert test_split1.equals(test_split2)

if __name__ == "__main__":
    import pytest
    import os
    _test_launch_steer()
    pytest.main([os.path.abspath(__file__), "-vv"])