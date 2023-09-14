from trace_model import *
from model_utils import *
from trace_utils import *
    
        
def trace_with_patch(
    model,
    prompts,
    heads_to_patch, # (head_index, layername)
    states_to_patch, # (layer_from, layer_to, start_tok, end_tok)
):
    pass

def noise_attn_heads(
    trace_model,
    prompt,
    heads_to_noise, # (head_index, layernum)
    noise = 0.1,
    replace=True,
    uniform_noise=False
):
    model = trace_model.model
    tokenizer = trace_model.tokenizer
    toks = tokenizer([prompt], padding=True, return_tensors="pt").to(model.device)
    inp = toks["input_ids"]
    
    ## noise create
    uniform_noise = False
    noise = 0.1
    replace=True
    rs = numpy.random.RandomState(1)  # For reproducibility, use pseudorandom noise
    if uniform_noise:
        prng = lambda *shape: rs.uniform(-1, 1, shape)
    else:
        prng = lambda *shape: rs.randn(*shape)
    if isinstance(noise, float):
        noise_fn = lambda x: noise * x
    else:
        noise_fn = noise
        
    heads_to_noise = [(head, layername(model, layer)) for (head, layer) in heads_to_noise]
    head_dim = model.config.n_embd // model.config.n_head
    
    def patch_rep(x, layer): # x is the output of the attn forward method (layer), layer is layer num
        ## x is tuple (attn_weights, present[key value tracker for MQA], opt(attn_weights))
        if layer in [layer for (_, layer) in heads_to_noise]:
            for head in [head for (head, layer_) in heads_to_noise if layer_ == layer]:
                # print("patching (layer, head): ", layer, head)
                ## noise at the corresponding heads
                start = head_dim * head
                end = start + head_dim
                noise_data = noise_fn(
                        torch.from_numpy(prng(x[0].shape[0], x[0].shape[1], head_dim))
                    ).to(x[0].device)
                if replace:
                    x[0][:,:,start:end] = noise_data
                else:
                    x[0][:,:,start:end] += noise_data
        return x


    with torch.no_grad(), TraceDict(
        model,
        [layername(model, i) for i in range(1, 40)],
        edit_output_attn=patch_rep,
    ) as trace_dict:
        model_out = model(inp)

    trace_dict.close()
    new_toks, logits, softmax = trace_decode(model_out, do_greedy_decoding=True, gather_only_topk_logits=model.config.vocab_size)
    txt = [tokenizer.decode(x) for x in new_toks.detach().cpu().numpy().tolist()]
    
    return txt

# def noise_hidden_states(
#     trace_model,
#     prompt,
#     heads_to_noise, # (head_index, layernum)
#     noise = 0.1,
#     replace=True,
#     uniform_noise=False
# ):
#     model = trace_model.model
#     tokenizer = trace_model.tokenizer
#     toks = tokenizer([prompt], padding=True, return_tensors="pt").to(model.device)
#     inp = toks["input_ids"]
    
#     ## noise create
#     uniform_noise = False
#     noise = 0.1
#     replace=True
#     rs = numpy.random.RandomState(1)  # For reproducibility, use pseudorandom noise
#     if uniform_noise:
#         prng = lambda *shape: rs.uniform(-1, 1, shape)
#     else:
#         prng = lambda *shape: rs.randn(*shape)
#     if isinstance(noise, float):
#         noise_fn = lambda x: noise * x
#     else:
#         noise_fn = noise
        
#     heads_to_noise = [(head, layername(model, layer)) for (head, layer) in heads_to_noise]
#     head_dim = model.config.n_embd // model.config.n_head
    
#     def patch_rep(x, layer): # x is the output of the attn forward method (layer), layer is layer num
#         ## x is tuple (attn_weights, present[key value tracker for MQA], opt(attn_weights))
#         if layer in [layer for (_, layer) in heads_to_noise]:
#             for head in [head for (head, layer_) in heads_to_noise if layer_ == layer]:
#                 # print("patching (layer, head): ", layer, head)
#                 ## noise at the corresponding heads
#                 start = head_dim * head
#                 end = start + head_dim
#                 noise_data = noise_fn(
#                         torch.from_numpy(prng(x[0].shape[0], x[0].shape[1], head_dim))
#                     ).to(x[0].device)
#                 if replace:
#                     x[0][:,:,start:end] = noise_data
#                 else:
#                     x[0][:,:,start:end] += noise_data
#         return x


#     with torch.no_grad(), TraceDict(
#         model,
#         [layername(model, i) for i in range(1, 40)],
#         edit_output_attn=patch_rep,
#     ) as trace_dict:
#         model_out = model(inp)

#     trace_dict.close()
#     new_toks, logits, softmax = trace_decode(model_out, do_greedy_decoding=True, gather_only_topk_logits=model.config.vocab_size)
#     txt = [tokenizer.decode(x) for x in new_toks.detach().cpu().numpy().tolist()]
    
#     return txt

def noise_hidden_states(
    trace_model,
    prompt,
    tok_range_per_state, # [(layer, start_tok, end_tok)]
    do_greedy_decoding= True,
    pick_from_topk = None,
    replace=True,
    uniform_noise=False,
    noise = 0.1
):
    model = trace_model.model
    tokenizer = trace_model.tokenizer
    prompt = [prompt]
    tok_range_per_state = [(layername(model, layer), start_tok, end_tok) 
                            for (layer, start_tok, end_tok) in tok_range_per_state]
    toks = tokenizer(prompt, padding=True, return_tensors="pt").to(model.device)
    inp = toks["input_ids"]
    embed_layername = layername(model, 0, "embed")
    
    # keep embed layer uncorrupted    
    assert embed_layername not in [layer for (layer, _, _) in tok_range_per_state]
    
    if pick_from_topk is None:
        pick_from_topk = model.config.vocab_size
        
    
    ## noise create
    uniform_noise = False
    noise = 0.1
    replace=True
    rs = numpy.random.RandomState(1)  # For reproducibility, use pseudorandom noise
    if uniform_noise:
        prng = lambda *shape: rs.uniform(-1, 1, shape)
    else:
        prng = lambda *shape: rs.randn(*shape)
    if isinstance(noise, float):
        noise_fn = lambda x: noise * x
    else:
        noise_fn = noise
        
    
    def patch_rep(x, layer): # x is the output of the layer
        # print(layer)
        if layer in [layer for (layer, _, _) in tok_range_per_state]:
            for (start, end) in [(start, end) for (layer_, start, end) in tok_range_per_state if layer_ == layer]:
                # print("patching (layer, head): ", layer, head)
                ## noise at the corresponding heads
                print(x[0].shape)
                noise_data = noise_fn(
                        torch.from_numpy(prng(x[0].shape[0], 1, x[0].shape[2]))
                    ).to(x[0].device)
                if replace:
                    x[0][:,start:end,:] = noise_data
                else:
                    x[0][:,start:end,:] += noise_data
        return x
        

    # With the patching rules defined, run the patched model in inference.
    with torch.no_grad(), TraceDict(
        model,
        [layername(model, i) for i in range(1, 40)],
        edit_output_block=patch_rep,
    ) as td:
        model_out = model(inp)

    td.close()
    new_toks, logits, probs = trace_decode(model_out, 
                                           do_greedy_decoding=do_greedy_decoding, 
                                           gather_only_topk_logits=pick_from_topk)
    txt = [tokenizer.decode(x) for x in new_toks.detach().cpu().numpy().tolist()]
    return txt
    
def patch_a_b_layers(
    trace_model,
    prompt_a,
    prompt_b,
    layera_to_layerb,
    pick_greedily=True,
    pick_from_topk = None,
):
    layers_dst = [layername(trace_model.model, i) for i in list(list(zip(*layera_to_layerb))[1])]
    layers_src = [layername(trace_model.model, i) for i in list(list(zip(*layera_to_layerb))[0])]
    
    dst_2_src= {layername(trace_model.model, to) : layername(trace_model.model, from_) 
                        for (from_, to) in layera_to_layerb}
    src_2_dst = {from_: to for to, from_ in dst_2_src.items()}
    
    if pick_from_topk is None:
        pick_from_topk = trace_model.model.config.vocab_size
        
    toks_a = trace_model.tokenizer(prompt_a, padding=True, return_tensors="pt").to(trace_model.model.device)
    inp_a = toks_a["input_ids"]
    
    embed_layername = layername(trace_model.model, 0, "embed")
    
    # keep embed layer uncorrupted
    
    assert embed_layername not in layers_dst+layers_src
    
    src_activations = {layername(trace_model.model, i): None for i in range(1, 40)}
    
    # run a
    def patch_rep_a(x, layer): # x is the output of the layer
        if layer in layers_src:
            src_activations[layer] = x
            assert all(torch.eq(src_activations[layer][0], x[0]).flatten().tolist())
        return x

    # run inf
    # With the patching rules defined, run the patched model in inference.
    with torch.no_grad(), TraceDict(
        trace_model.model,
        [layername(trace_model.model, i) for i in range(1, 40)],
        edit_output_block=patch_rep_a,
    ) as td:
        model_out = trace_model.model(inp_a)
        
    # run b
    def patch_rep_b(x, layer): # x is the output of the layer
        if layer in layers_dst:
            # print("DST", x, src_activations[dst_2_src[layer]])
            assert False in (torch.eq(src_activations[dst_2_src[layer]][0], x[0]).flatten().tolist())
            return src_activations[dst_2_src[layer]]
        return x

    toks_b = trace_model.tokenizer(prompt_b, padding=True, return_tensors="pt").to(trace_model.model.device)
    inp_b = toks_b["input_ids"]
    
    # run inf
    # With the patching rules defined, run the patched model in inference.
    with torch.no_grad(), TraceDict(
        trace_model.model,
        [layername(trace_model.model, i) for i in range(1, 40)],
        edit_output_block=patch_rep_b,
    ) as td:
        model_out = trace_model.model(inp_b)
        
    # We report softmax probabilities for the answers_t token predictions of interest.
    probs = torch.nn.functional.softmax(model_out.logits[:, -1, :], dim=1)
    tk = torch.topk(probs, pick_from_topk, dim=1).indices
    softmax_out_top_k = torch.gather(probs, 1, tk)
    softmax_out_top_k = softmax_out_top_k / softmax_out_top_k.sum(1)[:, None]

    if(pick_greedily == False):
        new_tok_indices = torch.multinomial(softmax_out_top_k, 1)
        new_toks = torch.gather(tk, 1, new_tok_indices)
    else:
        new_tok_indices = torch.topk(softmax_out_top_k, dim=1, k=1)
        new_toks = torch.gather(tk, 1, new_tok_indices.indices)

    return new_toks, probs
        
def move_hidden_states(
    trace_model,
    prompts,
    state_to_state, # [(from,to)]
    pick_greedily= False,
    pick_from_topk = None
):
    layers_dst = [layername(trace_model.model, i) for i in list(list(zip(*state_to_state))[1])]
    layers_src = [layername(trace_model.model, i) for i in list(list(zip(*state_to_state))[0])]
    
    dst_2_src= {layername(trace_model.model, to) : layername(trace_model.model, from_) 
                        for (from_, to) in state_to_state}
    src_2_dst = {from_: to for to, from_ in dst_2_src.items()}
    
    if pick_from_topk is None:
        pick_from_topk = trace_model.model.config.vocab_size
        
    toks = trace_model.tokenizer(prompts, padding=True, return_tensors="pt").to(trace_model.model.device)
    inp = toks["input_ids"]
    
    embed_layername = layername(trace_model.model, 0, "embed")
    
    # keep embed layer uncorrupted
    
    assert embed_layername not in layers_dst+layers_src
    

    src_activations = {layername(trace_model.model, i): None for i in range(1, 40)}
    # print(layers_dst, layers_src, dst_2_src, src_2_dst)
    def patch_rep(x, layer): # x is the output of the layer
        # print(layer)
        if layer in layers_src:
            src_activations[layer] = x
            # print("SRC", x, src_activations[layer])
            assert all(torch.eq(src_activations[layer][0], x[0]).flatten().tolist())
            return x
        elif layer in layers_dst:
            # print("DST", x, src_activations[dst_2_src[layer]])
            assert False in (torch.eq(src_activations[dst_2_src[layer]][0], x[0]).flatten().tolist())
            return src_activations[dst_2_src[layer]]
        else:
            return x
        return x
        

    # With the patching rules defined, run the patched model in inference.
    with torch.no_grad(), TraceDict(
        trace_model.model,
        [layername(trace_model.model, i) for i in range(1, 40)],
        edit_output_block=patch_rep,
    ) as td:
        model_out = trace_model.model(inp)

    # We report softmax probabilities for the answers_t token predictions of interest.
    probs = torch.nn.functional.softmax(model_out.logits[:, -1, :], dim=1)
    tk = torch.topk(probs, pick_from_topk, dim=1).indices
    softmax_out_top_k = torch.gather(probs, 1, tk)
    softmax_out_top_k = softmax_out_top_k / softmax_out_top_k.sum(1)[:, None]

    if(pick_greedily == False):
        new_tok_indices = torch.multinomial(softmax_out_top_k, 1)
        new_toks = torch.gather(tk, 1, new_tok_indices)
    else:
        new_tok_indices = torch.topk(softmax_out_top_k, dim=1, k=1)
        new_toks = torch.gather(tk, 1, new_tok_indices.indices)

    return new_toks, probs
    

      
    # # # def search_causal_heads(self, prompt, layers = range(20,31), replace=False, noise=0.9):
    # # #     heads_to_patch = []
    # # #     for l in layers:
    # # #         layername = trace_model.layername(l)
    # # #         heads_to_patch += [(i, layername) for i in range(48)]
            
    # # #     probs = trace_model.trace_with_patch(prompt, heads_to_patch=heads_to_patch, 
    # # #                             replace=replace, noise = noise)
    # # #     top_completion = trace_model.tokenizer.decode(probs.argmax(dim=0))
    # # #     # print(top_completion, heads_to_patch)
    # # #     try:
    # # #         tc = int(top_completion)
    # # #     except:
    # # #         return []
            
    # #     # return heads_to_patch