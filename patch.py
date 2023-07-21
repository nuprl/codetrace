
    
        
    def trace_with_patch(
        self,
        prompts,
        heads_to_patch, # (head_index, layername)
        states_to_patch, # (layer_from, layer_to, start_tok, end_tok)
    ):
        pass
    
    def noise_hidden_states(
        self,
        prompts,
        tok_range_per_state, # [(layer, start_tok, end_tok)]
        pick_greedily= False,
        pick_from_topk = None,
    ):
        tok_range_per_state = [(layername(self.model, layer), start_tok, end_tok) 
                               for (layer, start_tok, end_tok) in tok_range_per_state]
        
        if pick_from_topk is None:
            pick_from_topk = self.model.config.vocab_size
            
        toks = self.tokenizer(prompts, padding=True, return_tensors="pt").to(self.model.device)
        inp = toks["input_ids"]
        
        embed_layername = layername(self.model, 0, "embed")
        
        # keep embed layer uncorrupted    
        assert embed_layername not in [layer for (layer, _, _) in tok_range_per_state]
        

        src_activations = {layername(self.model, i): None for i in range(1, 40)}
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
            self.model,
            [layername(self.model, i) for i in range(1, 40)],
            edit_output_block=patch_rep,
        ) as td:
            model_out = self.model(inp)

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
        self,
        prompts,
        state_to_state, # [(from,to)] # start, end tok
        pick_greedily= False,
        pick_from_topk = None
    ):
        layers_dst = [layername(self.model, i) for i in list(list(zip(*state_to_state))[1])]
        layers_src = [layername(self.model, i) for i in list(list(zip(*state_to_state))[0])]
        
        dst_2_src= {layername(self.model, to) : layername(self.model, from_) 
                          for (from_, to) in state_to_state}
        src_2_dst = {from_: to for to, from_ in dst_2_src.items()}
        
        if pick_from_topk is None:
            pick_from_topk = self.model.config.vocab_size
            
        toks = self.tokenizer(prompts, padding=True, return_tensors="pt").to(self.model.device)
        inp = toks["input_ids"]
        
        embed_layername = layername(self.model, 0, "embed")
        
        # keep embed layer uncorrupted
        
        assert embed_layername not in layers_dst+layers_src
        

        src_activations = {layername(self.model, i): None for i in range(1, 40)}
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
            self.model,
            [layername(self.model, i) for i in range(1, 40)],
            edit_output_block=patch_rep,
        ) as td:
            model_out = self.model(inp)

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
    

      
    # def search_causal_heads(self, prompt, layers = range(20,31), replace=False, noise=0.9):
    #     heads_to_patch = []
    #     for l in layers:
    #         layername = self.layername(l)
    #         heads_to_patch += [(i, layername) for i in range(48)]
            
    #     probs = self.trace_with_patch(prompt, heads_to_patch=heads_to_patch, 
    #                             replace=replace, noise = noise)
    #     top_completion = self.tokenizer.decode(probs.argmax(dim=0))
    #     # print(top_completion, heads_to_patch)
    #     try:
    #         tc = int(top_completion)
    #     except:
    #         return []
            
        # return heads_to_patch