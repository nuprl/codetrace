import torch
from trace_utils import *
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList
from typing import Union, List
from model_utils import extract_layer_formats, layername
import numpy as np
import unicodedata
from pathlib import Path
import os
import numpy
from collections import defaultdict
# from baukit.nethook import get_module, set_requires_grad
from model_utils import *
'''

'''


class ModelLoader:
    def __init__(self, 
                 model_name_or_path, 
                 dtype = torch.float32) -> None:
        
        self.model_name = model_name_or_path
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            output_attentions=True,
            use_auth_token=True,
            low_cpu_mem_usage=True, ## loads with accelerate
            torch_dtype=dtype,
            trust_remote_code=True,
            
        )
        torch.set_grad_enabled(False)
        self.model.eval().cuda()
         
        # post process
        self.extract_fields()
        
        set_requires_grad(False, self.model)
        
        ## set pad tokens
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name) 
        self.tokenizer.clean_up_tokenization_spaces=False
        self.tokenizer.padding_side="left"
        self.tokenizer.pad_token = self.tokenizer.eos_token  
         
    def generate(self, prompt, max_new_tokens=100, stop_tokens = [], temperature=1.0, only_generated=False, do_print=False, do_sample=False):
        inputs = self.tokenizer(prompt, padding=True, return_tensors="pt").to(self.model.device)
        ## stop criteria
        stop_ids = [self.tokenizer(stop_tok, padding=True, return_tensors='pt')['input_ids'] 
                    for stop_tok in stop_tokens]
        
        encounters = [prompt.count(stop_tok)+1 for stop_tok in stop_tokens]
        
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_ids, 
                                                                        encounters=encounters)])
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                         pad_token_id=self.tokenizer.eos_token_id,
                                         stopping_criteria=stopping_criteria,
                                         temperature=temperature,
                                         do_sample=do_sample)
        
        generated = self.tokenizer.decode(outputs[0])[len(prompt):]
        
        if only_generated:
            if do_print:
                print(generated)
            return generated
        else:
            if do_print:
                print(prompt + generated)
            return prompt + generated
    
    def extract_fields(self):
        """
        model_type
        no_split_module_classes
        layer_name_format
        mlp_module_name_format
        attn_module_name_format
        """
        self.model_type = self.model.config.model_type
        self.no_split_module_classes = self.model._no_split_modules
       
        formats = extract_layer_formats(self.model.named_parameters)
        self.layer_name_format = formats["layer"]
        self.mlp_module_name_format = formats["mlp"]
        self.attn_module_name_format = formats["attn"]
        

        if(self.model_type is not None):
            self.layer_names = [self.layer_name_format.format(i) for i in range(self.model.config.n_layer)]
            self.mlp_module_names = [self.mlp_module_name_format.format(i) for i in range(self.model.config.n_layer)]
            self.attn_module_names = [self.attn_module_name_format.format(i) for i in range(self.model.config.n_layer)]
            self.tracable_modules =  self.mlp_module_names + self.attn_module_names + self.layer_names
            
    def trace_generate(
            self,
            prompts: Union[str, List[str]],                 
            max_new_toks: int = 1,          
            request_activations: List[int] = [],
            request_logits: List[int] = [],
            report_topk_logits : int = 10,
            pick_from_topk : int = None,
            pick_greedily : bool = False, 
        ):
        '''
        Trace with cache implementation if possible
        '''
        if(type(prompts) == str):
            prompts = [prompts]
        
        # print(self.tracable_modules)
        if(len(request_activations) > 0):
            invalid_module = list(set(request_activations) - set(self.tracable_modules))
            assert(
                len(invalid_module) == 0
            ), f"modules {invalid_module} are not in the list of tracable modules"
            activation_track = {k: None for k in request_activations}
            attn_track = {k: None for k in request_activations}
            mlp_track = {k: None for k in request_activations}
        if(len(request_logits) > 0):
            invalid_module = list(set(request_logits) - set(self.tracable_modules))
            assert(
                len(invalid_module) == 0
            ), f"modules {invalid_module} are not in the list of tracable modules"
            layer_logits_track = {k: None for k in request_logits}
        
        tokenized = self.tokenizer(prompts, padding=True, return_tensors="pt").to(self.model.device)
        input_ids, attention_mask = tokenized["input_ids"], tokenized["attention_mask"]
        prompt_len = input_ids.size(1)
        batch_size = input_ids.size(0)
        assert batch_size == 1, "batch size must be 1"

        ## add prompt to ret dict
        report_input_tokenized = []
        for b in range(batch_size):
            curr_inp_tok = []
            for t, a in zip(input_ids[b], attention_mask[b]):
                if(a == 0):
                    break
                curr_inp_tok.append((self.tokenizer.decode(t), t.item()))
            report_input_tokenized.append((curr_inp_tok))
        ret_dict = {"input_tokenized": report_input_tokenized}


        # init size of context
        past_key_values, cur_context = None, slice(0, attention_mask.sum(1).min().item()) 

        
        generated_tokens = [[] for _ in range(input_ids.size(0))]
        with torch.no_grad():
            max_out_len =  prompt_len + max_new_toks
            while input_ids.size(1) < max_out_len: 

                ## traces curr_inputs -> prediction of next tok
                with TraceDict(
                    self.model, 
                    layers = request_activations+request_logits,
                    retain_input=True,
                    retain_output=True
                ) as traces:
                    model_out = self.model(
                        input_ids=input_ids[:, cur_context],
                        attention_mask=attention_mask[:, cur_context],
                        past_key_values=past_key_values,
                        use_cache = False,  
                    )

                if(len(request_activations) > 0):
                    assert layername(self.model, 0, "embed") not in request_activations, "Embedding layer is not supported"
                    if(input_ids.size(1) == max_out_len - 1):
                        for module in request_activations:
                            if(activation_track[module] is None):
                                activation_track[module] = traces[module].block_output[0].cpu().numpy()
                            if(attn_track[module] is None):
                                attn_track[module] = traces[module].attn_output[0].cpu().numpy()
                            if(mlp_track[module] is None):
                                mlp_track[module] = traces[module].mlp_output[0].cpu().numpy()
                if(len(request_logits) > 0):
                    assert layername(self.model, 0, "embed") not in request_logits, "Embedding layer is not supported"
                    if(input_ids.size(1) == max_out_len - 1):
                        for module in request_logits:
                            if(layer_logits_track[module] is None):
                                probs = traces[module].block_output[0]
                                lm_head = get_module(self.model, "lm_head")
                                ln_f = get_module(self.model, "transformer.ln_f")
                                apply_head = torch.softmax(
                                        lm_head(ln_f(probs[:, -1, :])), dim=1
                                    )
                                rets = torch.topk(apply_head[-1], report_topk_logits)
                                tokenized = [self.tokenizer.decode(i) for i in rets.indices]
                                assert(len(tokenized) == report_topk_logits)
                                layer_logits_track[module] = list(zip(tokenized, rets.values.cpu().numpy()))
                                
                final_logits, past_key_values = model_out.logits, model_out.past_key_values

                softmax_out = torch.nn.functional.softmax(final_logits[:, -1, :], dim=1)

                # Top-k sampling
                if pick_from_topk == None:
                    pick_from_topk = self.model.config.vocab_size
                tk = torch.topk(softmax_out, pick_from_topk, dim=1).indices
                softmax_out_top_k = torch.gather(softmax_out, 1, tk)
                softmax_out_top_k = softmax_out_top_k / softmax_out_top_k.sum(1)[:, None]

                if(pick_greedily == False):
                    new_tok_indices = torch.multinomial(softmax_out_top_k, 1)
                    new_toks = torch.gather(tk, 1, new_tok_indices)
                else:
                    new_tok_indices = torch.topk(softmax_out_top_k, dim=1, k=1)
                    new_toks = torch.gather(tk, 1, new_tok_indices.indices)

                for i in range(input_ids.size(0)):
                    generated_tokens[i].append(
                        [
                            {"token": self.tokenizer.decode(t), "id": t.item(), "p": softmax_out[i][t.item()].item()}
                            for t in tk[i]
                        ]
                    )


                ## Auto-regression: insert new tok into inputs

                # If we're currently generating the continuation for the last token in `input_ids`,
                # create a new index so we can insert the new token
                if cur_context.stop == input_ids.size(1):
                    attention_mask = torch.cat(
                        [attention_mask, attention_mask.new_zeros(batch_size, 1)], dim=1
                    )
                    input_ids = torch.cat(
                        [
                            input_ids,
                            input_ids.new_ones(batch_size, 1) * self.tokenizer.pad_token_id,
                        ],
                        dim=1,
                    )

                ## check if new generation idx is out of bounds of max_len
                last_non_masked = attention_mask.sum(1) - 1   ## idx of last 1 in attn_mask

                for i in range(batch_size):
                    new_idx = last_non_masked[i] + 1 ## idx of new tok

                    ## if new_idx not context end (no generation output)
                    if last_non_masked[i].item() + 1 != cur_context.stop:
                        continue


                    # Stop generating if we've already maxed out for this prompt
                    if new_idx < max_out_len:
                        # print("new_toks", new_toks[i], i)
                        input_ids[i][new_idx] = new_toks[i]
                        attention_mask[i][new_idx] = 1
                        cur_context = slice(0, new_idx.item()+1)

            ## End gen loop:

            cur_context = slice(0, cur_context.stop + 1)

            # clear up GPU
            del(traces)
            del(model_out)
            torch.cuda.empty_cache()                

            txt = [self.tokenizer.decode(x) for x in input_ids.detach().cpu().numpy().tolist()]

            txt = [
                unicodedata.normalize("NFKD", x)
                # .replace("\n\n", " ")
                # .replace("<|endoftext|>", "")
                for x in txt
            ]

            ret_dict["generated_tokens"] = generated_tokens
            if(request_activations is not None and len(request_activations) > 0):
                ret_dict["block"] = activation_track
                ret_dict["attn"] = attn_track
                ret_dict["mlp"] = mlp_track
            if request_logits is not None and len(request_logits) > 0:
                ret_dict["logits"] = layer_logits_track
            return txt, ret_dict

    
        
    def trace_with_patch(
        self,
        prompts,
        heads_to_patch, # (head_index, layername)
        states_to_patch, # (layer_from, layer_to, start_tok, end_tok)
    ):
        pass
    
        
    def patch_hidden_states(
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