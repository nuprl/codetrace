from trace_model import *
from model_utils import *
from trace_utils import *
from typing import Tuple, List, Dict, Union, Callable
from collections import defaultdict
import torch
import numpy
"""
TODO: 

- [X] Make noise_patching more efficient 
use numpy fast matrix ops instead of for-loops

- [] debug

- [] add batching

- [] add attn knockout

"""
    

class TraceModel(TraceBase):
    
    # TODO: debug this
    def noise_attn_heads(
        self,
        prompt : str,
        heads_to_noise : List[Tuple[int, int]], # (head_index, layernum)
        noise_fn : Callable,
        replace_with_noise : bool = False,
        do_greedy_decoding : bool = True,
        gather_only_topk_logits : int = None,
    ):
        model = self.model
        head_dim = model.config.n_embd // model.config.n_head
        
        target_layers = []
        layer_to_indices = defaultdict(list)
        request_activations = []
        request_logits = []
        for h,layer in heads_to_noise:
            l = self.layername(layer)
            target_layers.append(l)
            start = head_dim * h
            end = start + head_dim
            indices = list(range(start, end))
            layer_to_indices[l] += indices
            request_activations.append(l)
            request_logits.append(l)
            
        def patch_rep(x, layer): 
            # x is the output of the attn forward method (layer), layer is layer num
            ## x is tuple (attn_weights, present[key value tracker for MQA], opt(attn_weights))
            if layer in target_layers:
                print(f"Visited layer {layer}...")
                saved_x = x[0].cpu().numpy()
                if replace_with_noise:
                    x[0][:,:,layer_to_indices[layer]] = noise_fn(x[0][:,:,layer_to_indices[layer]].shape).to(x[0].device)
                else:
                    x[0][:,:,layer_to_indices[layer]] += noise_fn(x[0][:,:,layer_to_indices[layer]].shape).to(x[0].device)
                assert not numpy.all(numpy.equal(x[0].cpu().numpy(), saved_x))
            return x

        return self.trace_generate(prompt, 
                                request_activations=request_activations,
                                request_logits=request_logits,
                                edit_output_attn=patch_rep,
                                do_greedy_decoding=do_greedy_decoding,
                                gather_only_topk_logits=gather_only_topk_logits)



    def noise_hidden_states(
        self,
        prompt : str,
        tok_range_per_state : List[Tuple[int, List[int]]], # [(layer, [tok_indices_to_noise])]
        noise_fn : Callable,
        replace_with_noise : bool = False,
        do_greedy_decoding : bool = True,
        gather_only_topk_logits : int = None,
    ):
        model = self.model
        tokenizer = self.tokenizer
        
        target_layers = []
        request_activations = []
        request_logits = []
        layer_to_indices = defaultdict(list)
        for (layer, indices) in tok_range_per_state:
            layer = self.layername(layer)
            target_layers.append(layer)
            request_activations.append(layer)
            request_logits.append(layer)
            layer_to_indices[layer] += indices
        
        def patch_rep(x, layer): # x is the output of the layer
            if layer in target_layers:
                print(f"Visited layer {layer}...")
                saved_x = x[0].cpu().numpy()
                if replace_with_noise:
                    x[0][:,layer_to_indices[layer],:] = noise_fn(x[0][:,layer_to_indices[layer],:].shape).to(x[0].device)
                else:
                    x[0][:,layer_to_indices[layer],:] += noise_fn(x[0][:,layer_to_indices[layer],:].shape).to(x[0].device)
                assert not numpy.all(numpy.equal(x[0].cpu().numpy(), saved_x))
            return x
            
        return self.trace_generate(prompt, 
                                   request_activations=request_activations,
                                   request_logits=request_logits,
                                   edit_output_block=patch_rep,
                                   do_greedy_decoding=do_greedy_decoding,
                                   gather_only_topk_logits=gather_only_topk_logits)
        
    # TODO: debug this
    def patch_layer_prompt_a_to_prompt_b(
        self,
        prompt_a : str,
        prompt_b : str,
        layer_a_to_b : List[Tuple[int, int]], # [(from, to)]
        do_greedy_decoding=True,
        gather_only_topk_logits= None,
    ):
        model = self.model
        tokenizer = self.tokenizer
        
        layers_dst = []
        layers_src = []
        dst_2_src= {}
        src_2_dst = {}
        for (src, dst) in layer_a_to_b:
            src = self.layername(src)
            dst = self.layername(dst)
            layers_src.append(src)
            layers_dst.append(dst)
            dst_2_src[dst] = src
            src_2_dst[src] = dst
        
        toks_a = tokenizer(prompt_a, padding=True, return_tensors="pt").to(model.device)
        inp_a = toks_a["input_ids"]
        
        src_activations = {layername(model, i): None for i in range(1, 40)}
        
        # run a: collect layer from a, and nothing else
        def patch_rep_a(x, layer):
            if layer in layers_src:
                print("Visiting layer ", layer)
                src_activations[layer] = x
                # assert all(torch.eq(src_activations[layer][0], x[0]).flatten().tolist())
            return x

        # run the patched model in inference.
        with torch.no_grad(), TraceDict(
            model,
            layers = list(set(layers_src)),
            edit_output_block=patch_rep_a,
        ) as td:
            model_out = model(inp_a)
        td.close()
            
        # run b: collect all desired params 
        def patch_rep_b(x, layer): # x is the output of the layer
            print("Visiting layer ", layer)
            if layer in layers_dst:
                # assert False in (torch.eq(src_activations[dst_2_src[layer]][0], x[0]).flatten().tolist())
                return src_activations[dst_2_src[layer]]
            return x

        return self.trace_generate(prompt_b,
                                    request_activations=list(set(layers_dst)),
                                    request_logits=list(set(layers_dst)),
                                    edit_output_block=patch_rep_b,
                                    do_greedy_decoding=do_greedy_decoding,
                                    gather_only_topk_logits=gather_only_topk_logits)
        

    def patch_layer_to_layer(
        self,
        prompt : str,
        layer_a_to_b : List[Tuple[int, int]], # [(from,to)]
        request_activations : List[str] = [],
        request_logits : List[str] = [],
        do_greedy_decoding=True,
        gather_only_topk_logits= None
    ):
        model = self.model
        tokenizer = self.tokenizer
        
        layers_dst = []
        layers_src = []
        dst_2_src= {}
        src_2_dst = {}
        for (src, dst) in layer_a_to_b:
            src = self.layername(src)
            dst = self.layername(dst)
            layers_src.append(src)
            layers_dst.append(dst)
            dst_2_src[dst] = src
            src_2_dst[src] = dst
        
        toks = tokenizer(prompt, padding=True, return_tensors="pt").to(model.device)
        inp = toks["input_ids"]
        
        target_layers = list(set(layers_dst + layers_src))
        
        src_activations = {layername(model, i): None for i in range(1, 40)}

        def patch_rep(x, layer): # x is the output of the layer
            print("Visiting layer  :", layer)
            if layer in layers_src:
                src_activations[layer] = x
                assert all(torch.eq(src_activations[layer][0], x[0]).flatten().tolist())
                return x
            elif layer in layers_dst:
                assert False in (torch.eq(src_activations[dst_2_src[layer]][0], x[0]).flatten().tolist())
                return src_activations[dst_2_src[layer]]
            else:
                return x

        return self.trace_generate(prompt,
                                    request_activations=target_layers,
                                    request_logits=target_layers,
                                    edit_output_block=patch_rep,
                                    do_greedy_decoding=do_greedy_decoding,
                                    gather_only_topk_logits=gather_only_topk_logits)
    