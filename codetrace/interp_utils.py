"""
Interp utils
"""
import torch
from nnsight import LanguageModel
import nnsight
from nnsight.tracing.Proxy import Proxy
from typing import List, Union, Callable, Optional, TypeVar
import transformers
from torchtyping import TensorType
from codetrace.utils import (
    topk_filtering,
    pos_indexing,
    masked_get,
    masked_fill,
    lm_decode,
    get_lm_layers,
    apply_reduction
)

"""
Define some tensor types
"""
ActivationTensor = TypeVar(TensorType[float,"n_layer","n_prompt","n_tokens","hdim"])
ReducedActivationTensor = TypeVar(TensorType[float,"n_layer","n_prompt","hdim"])
LayerActivationTensor = TypeVar(TensorType[float, "n_prompt","n_tokens","hdim"])

MaskTensor = TypeVar(TensorType[bool,"n_layer","n_prompt","n_tokens","hdim"])

IndicesTensor = TypeVar(TensorType[int, "n_layer","n_prompt","n_tokens","top_k"])
ProbabilitiesTensor = TypeVar(TensorType[float, "n_layer","n_prompt","n_tokens","top_k"], )

class LogitResult:
    """
    Wrapper over logits and token indices.
    Tensors in shape: [n_layer, n_prompt, n_tokens, topk]
    """
    def __init__(self, token_indices : IndicesTensor, probabilities : ProbabilitiesTensor):
        self.token_indices = token_indices
        self.probabilities = probabilities
        
    def __getitem__(self, idx:int) -> "LogitResult":
        return LogitResult(self.token_indices[idx], self.probabilities[idx])
    
    def tokens(self, tokenizer : transformers.PreTrainedTokenizer) -> List[str]:
        """
        Decode tokens to strings
        """
        if self.token_indices.squeeze().dim() > 1:
            raise ValueError(f"token_indices must be 1d")

        return tokenizer.decode(self.token_indices.flatten().numpy())

class TraceResult:
    """
    Wrapper over hidden states and logits.
    
    Arguments:
        logits : torch.Tensor
            logits from model, have shape [n_layer, n_prompt, n_tokens, n_vocab]
        layer_idxs : List[int]
            from which model layers the logits were collected
        model_n_layer : int
            number of layers in model
        hidden_states : torch.Tensor
            hidden states from model, have shape [n_layer, n_prompt, n_tokens, n_embd]
        custom_decoder : Union[None, torch.nn.Module]
            custom decoder to apply to logits
    
    NOTE:
    - assumes logit input is unsoftmaxed
    - all tensors that get passed go through are detached and sent to cpu
    - input/output tensors are stacked
    
    A concern with this class is OOM errors. Temporary solution is to detach and move all tensors to cpu.
    can cause RAM errors if too many tensors are stored.
    """
    
    def __init__(
        self, 
        logits : ActivationTensor, 
        layer_idxs : List[int],
        model_n_layer : int,
        hidden_states : ActivationTensor = None,
        custom_decoder : Union[None, torch.nn.Module] = None
    ):
        self._logits = logits.detach().cpu()
        if hidden_states != None:
            hidden_states = hidden_states.detach().cpu()
        self._hidden_states = hidden_states
        del(hidden_states)
        del(logits)
        self.n_layers = model_n_layer
        # if layer idxs are negative indexing, convert to positive index
        self._layer_idx = [pos_indexing(i, n=model_n_layer) for i in layer_idxs]
        self.custom_decoder = custom_decoder
        
    def decode_logits(
        self, 
        prompt_idx : List[int],
        token_idx : List[int],
        top_k : int = 1,
        layers : List[int] = [-1],
        do_log_probs : bool = False,
    ) -> LogitResult:
        """
        Decode logits to tokens, after scoring top_k.
        NOTE: layers will index the literal layer index, not the index in the list. This means
        -1 corresponds to the last layer model_n_layer - 1.
        """
        # following will throw error if index is not present in collected layer logits:
        layers = [self._layer_idx.index(pos_indexing(i, n=self.n_layers)) for i in layers]
        # get ind
        token_idx = [pos_indexing(i, self._logits.shape[2]) for i in token_idx]

        # select logits while maintaining len(shape) = 4
        logits = self._logits[layers][:,prompt_idx][:,:,token_idx]
        
        if self.custom_decoder is not None:
            logits = self.custom_decoder(logits)
        
        logits = topk_filtering(logits, top_k, do_log_probs)
        return LogitResult(logits.indices, logits.values)

@torch.no_grad
def collect_hidden_states(
    model : LanguageModel,
    prompts : List[str],
    layers : List[int] = None,
    target_fn : Optional[Callable[[ActivationTensor],MaskTensor]] = None,
    reduction: Optional[Union[str, Callable[[ActivationTensor,List[int]],ReducedActivationTensor]]]= None
) -> Union[ReducedActivationTensor, ActivationTensor]:
    """
    Collect hidden states for each prompt. 
    Optionally, collect hidden states at specific layers, or at certain
    token positions given by target_fn.

    Arguments:
        model: LM to collect from
        prompts: prompts to collect from
        layers: layers to collect from
        target_fn: a function that takes an activation and produces a mask over the indices
            indicating which indices should be collected and which not.
        reduction: provide an einops reduction function for reducing the collected activation to
            save gpu memory.
    """
    if layers is None:
        layers = list(range(len(get_lm_layers(model))))

    with model.trace() as tracer:
        with tracer.invoke(prompts) as invoker:
            hidden_states = []
            for layer_idx in layers:
                # output is a tuple of (hidden_states, present)
                hs = get_lm_layers(model)[layer_idx].output[0]
                if target_fn != None:
                    # mask shape: [n_prompts, n_toks]
                    mask = target_fn(invoker.inputs[0]["input_ids"])
                    hs = masked_get(hs, mask)
                    if reduction:
                        hs = apply_reduction(hs, reduction, dim=1)
                hidden_states.append(hs)

        hidden_states = torch.stack(hidden_states, dim=0).save() 
            
    hidden_states = nnsight.util.apply(hidden_states, lambda x: x.value, Proxy)
    return hidden_states

@torch.no_grad
def _prepare_patch(patch:Union[ReducedActivationTensor,ActivationTensor], n_tokens:int)->LayerActivationTensor:
    if patch.ndim == 4:
        return patch
    else:
        # pad left side with 0's s.t. n_tokens_to_patch == n_tokens
        return torch.nn.functional.pad(patch, (0,0,n_tokens - patch.shape[1],0))

@torch.no_grad
def insert_patch(
    model : LanguageModel,
    prompts : List[str],
    patch : Union[ReducedActivationTensor,ActivationTensor],
    layers_to_patch : List[int],
    target_fn : Optional[Callable[[ActivationTensor],MaskTensor]] = None,
    collect_hidden_states : bool = True,
    patch_fn: Optional[Callable[[ActivationTensor, MaskTensor, ActivationTensor],MaskTensor]] = None,
) -> TraceResult:
    """
    Insert patch at layers and tokens
    """
    if not patch_fn:
        patch_fn = masked_fill

    if collect_hidden_states:
        collect_range = list(range(len(get_lm_layers(model))))
    else:
        collect_range = [len(get_lm_layers(model))-1]

    patch = patch.to(model.device)
    with model.trace() as tracer:
        with tracer.invoke(prompts) as invoker:
            inputs = invoker.inputs[0]["input_ids"]
            mask = target_fn(inputs)
            hidden_states = []
            # iter over all layers
            for layer in range(len(get_lm_layers(model))):
                hs = get_lm_layers(model)[layer].output[0]
                
                if layer in layers_to_patch:
                    layer_patch = _prepare_patch(patch[layer], mask.shape[1]).to(model.device)
                    patched_hs = patch_fn(hs, mask.to(model.device), layer_patch)
                    for prompt_idx in range(hs.shape[0]):
                        get_lm_layers(model)[layer].output[0][prompt_idx,:,:] = patched_hs[prompt_idx,:]

                if layer in collect_range:
                    hidden_states.append(get_lm_layers(model)[layer].output[0])

            hidden_states = torch.stack(hidden_states,dim=0).save()
            logits = lm_decode(model, hidden_states, True).save()
            final_logits = model.lm_head.output.save()
    
    hidden_states = nnsight.util.apply(hidden_states, lambda x: x.value, Proxy)
    logits = nnsight.util.apply(logits, lambda x: x.value, Proxy)
    final_logits = nnsight.util.apply(final_logits, lambda x: x.value, Proxy)
    assert torch.equal(logits[-1], final_logits), f"{logits[-1]} != {final_logits}"
    return TraceResult(logits, collect_range, len(get_lm_layers(model)), hidden_states=hidden_states)

@torch.no_grad    
def logit_lens(
    model : LanguageModel,
    prompts : List[str],
    layers : List[int] = None,
    store_hidden_states : bool = False
) -> TraceResult:
    """
    Apply logit lens to prompts
    """
    if layers is None:
        layers = list(range(len(get_lm_layers(model))))
    else:
        layers = [pos_indexing(x, len(get_lm_layers(model))) for x in layers]
    
    with model.trace() as tracer:
        with tracer.invoke(prompts) as invoker:
            hidden_states = torch.stack([
                get_lm_layers(model)[layer_idx].output[0]
                for layer_idx in layers
            ],dim=0).save()
            
            logits = lm_decode(model, hidden_states, True).save()
            final_logits = model.lm_head.output.save()

    logits = nnsight.util.apply(logits, lambda x: x.value, Proxy)
    final_logits = nnsight.util.apply(final_logits, lambda x: x.value, Proxy)
    assert torch.equal(logits[-1], final_logits)
    
    if store_hidden_states:
        hidden_states = nnsight.util.apply(hidden_states, lambda x: x.value, Proxy)
        return TraceResult(logits, layers, len(get_lm_layers(model)), hidden_states=hidden_states)
    else: 
        return TraceResult(logits, layers, model_n_layer=len(get_lm_layers(model)))
    
@torch.no_grad
def custom_lens(
    model : LanguageModel,
    decoder : torch.nn.Module,
    prompts : List[str],
    layers : List[int],
    activations : Optional[ActivationTensor] = None,
    k : int = 1
) -> List[str]:
    """
    Apply custom lens to model activations for prompt at (layer, token)
    Note: for original decoder, load transformer version of model and copy
    """
    if activations is None:
        activations = collect_hidden_states(model,prompts,layers=layers)
    else:
        activations = activations[layers]
    
    # apply decoder
    activations = activations.detach().cpu()
    logits = decoder(activations)

    # softmax
    logits = logits.softmax(dim=-1)
    topk= logits.topk(k, dim=-1)
    # output shape is [layer, prompt, tokens, vocab]
    return topk, activations, logits