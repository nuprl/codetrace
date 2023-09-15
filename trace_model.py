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
import time
from collections import defaultdict
from model_utils import *
'''
Wrapper class for Transformer models
'''

class TraceBase:
    def __init__(self, 
                 model_name_or_path, 
                 dtype = torch.float32) -> None:
        self.model_name = model_name_or_path
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            output_attentions=True,
            low_cpu_mem_usage=True, ## loads with accelerate
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        torch.set_grad_enabled(False)
        self.model.eval().cuda()
         
        # extract field names
        self.extract_fields()
        
        ## init tok
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name) 
        self.tokenizer.clean_up_tokenization_spaces=False
        self.tokenizer.padding_side="left"
        self.tokenizer.pad_token = self.tokenizer.eos_token  
    
    
    def layername(self, num : int, kind : str = None) -> str:
        """ Get layername """
        model = self.model
        if hasattr(model, "transformer"):
            if kind == "embed":
                return "transformer.wte"
            return f'transformer.h.{num}{"" if kind is None else "." + kind}'
        if hasattr(model, "gpt_neox"):
            if kind == "embed":
                return "gpt_neox.embed_in"
            if kind == "attn":
                kind = "attention"
            return f'gpt_neox.layers.{num}{"" if kind is None else "." + kind}'
        assert False, "unknown transformer structure"  
        
        
    def generate(self, 
                 prompt : str, 
                 max_new_tokens : int = 100, 
                 stop_tokens : List[str]= [], 
                 temperature : float=1.0, 
                 do_sample : bool =False,
                 append_prompt_prefix : bool = True):
        inputs = self.tokenizer(prompt, padding=True, return_tensors="pt").to(self.model.device)
        ## stop criteria
        stop_ids = [self.tokenizer(stop_tok, padding=True, return_tensors='pt')['input_ids'] 
                    for stop_tok in stop_tokens]
        
        encounters = [prompt.count(stop_tok)+1 for stop_tok in stop_tokens]
        
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_ids, 
                                                                        encounters=encounters)])
        outputs = self.model.generate(**inputs, 
                                         max_new_tokens=max_new_tokens,
                                         pad_token_id=self.tokenizer.eos_token_id,
                                         stopping_criteria=stopping_criteria,
                                         temperature=temperature,
                                         do_sample=do_sample)
        
        generated = self.tokenizer.decode(outputs[0])[len(prompt):]
        
        if append_prompt_prefix:
            return prompt + generated
        else:
            return generated
    
    def extract_fields(self):
        """
        Inits the following fields:
        - tracable_modules
        - model_type
        - no_split_module_classes
        - layer_name_format
        - mlp_module_name_format
        - attn_module_name_format
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
            prompt: str,                 
            max_new_toks: int = 512,
            stop_toks : List[str] = None,          
            request_activations: List[str] = [],
            request_logits: List[str] = [],
            do_greedy_decoding : bool = True,
            gather_only_topk_logits : int = None,
            edit_output_attn: callable = None,
            edit_output_mlp: callable = None,
            edit_output_block: callable = None,
        ):
        '''
        generate with trace (collecting either logits or activations at specified layers)
        '''
        if gather_only_topk_logits is None:
            # get all logits
            gather_only_topk_logits = self.model.config.vocab_size
            
        # Setup with warnings
        if max_new_toks is None and stop_toks is None:
            raise(ValueError, "Either max_new_toks or stop_toks must be specified")
        
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
        
        prompts = [prompt]
        tokenized = self.tokenizer(prompts, padding=True, return_tensors="pt").to(self.model.device)
        input_ids, attention_mask = tokenized["input_ids"], tokenized["attention_mask"]
        prompt_len = input_ids.size(1)
        batch_size = input_ids.size(0)
        assert batch_size == 1, "batch size must be 1" # TODO expand

        ## add prompt to ret dict
        prompt_toks = []
        for tok, mask in zip(input_ids[0], attention_mask[0]):
            if(mask == 0):
                break
            prompt_toks.append((self.tokenizer.decode(tok), tok.item()))
        ret_dict = {"input_tokenized": [prompt_toks]}

        # init size of context
        past_key_values, cur_context = None, slice(0, attention_mask.sum(1).min().item()) 
        generated_tokens = [[] for _ in range(input_ids.size(0))]
        
        # prep stopping criterias
        max_out_len = prompt_len + max_new_toks
            
        if stop_toks:
            stop_ids = [self.tokenizer(stop_tok, padding=True, return_tensors='pt')['input_ids'].tolist()[0]
                for stop_tok in stop_toks]
        else:
            stop_ids = [self.model.config.eos_token_id]
                
        # generate loop
        with torch.no_grad():
            # while < max_out_len and not stop_tok
            while input_ids.size(1) < max_out_len:
                # and input_ids[:, cur_context][:,-1].tolist() not in stop_ids):
                ## traces curr_inputs -> prediction of next tok
                with TraceDict(
                    self.model, 
                    layers = list(set(request_activations+request_logits)),
                    retain_input=True,
                    retain_output=True,
                    edit_output_attn=edit_output_attn,
                    edit_output_mlp=edit_output_mlp,
                    edit_output_block=edit_output_block,
                ) as traces:
                    model_out = self.model(
                        input_ids=input_ids[:, cur_context],
                        attention_mask=attention_mask[:, cur_context],
                        past_key_values=past_key_values,
                        use_cache = False,  
                    )
                traces.close()

                # collect activations
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
                # collect logits
                if(len(request_logits) > 0):
                    num_logits_to_return = 10
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
                                rets = torch.topk(apply_head[-1], num_logits_to_return)
                                tokenized = [self.tokenizer.decode(i) for i in rets.indices]
                                assert(len(tokenized) == num_logits_to_return)
                                layer_logits_track[module] = list(zip(tokenized, rets.values.cpu().numpy()))
                           
                new_toks, topk_logits, softmax_out= trace_decode(model_out, do_greedy_decoding, gather_only_topk_logits)


                def calc_probs(batch_i, logit_idx):
                    return softmax_out[batch_i][logit_idx].item()
                    
                # collect tokens generated from final layer
                # for every prompt in batch i 
                for batch_i in range(input_ids.size(0)): 
                    generated_tokens[batch_i].append(
                        [
                            {"token": self.tokenizer.decode(token_id), "id": token_id.item(), "p": calc_probs(batch_i, logit_idx)}
                            for logit_idx,token_id in enumerate(topk_logits[batch_i])
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
                    # and new_toks[i].tolist() not in stop_ids:
                        input_ids[i][new_idx] = new_toks[i]
                        attention_mask[i][new_idx] = 1
                        cur_context = slice(0, new_idx.item()+1)

            ## End gen loop:

            cur_context = slice(0, cur_context.stop + 1)

            # clear up GPU
            # del(traces)
            # del(model_out)
            torch.cuda.empty_cache()                

            txt = [self.tokenizer.decode(x) for x in input_ids.detach().cpu().numpy().tolist()]

            ret_dict["generated_tokens"] = generated_tokens
            if(request_activations is not None and len(request_activations) > 0):
                ret_dict["block"] = activation_track
                ret_dict["attn"] = attn_track
                ret_dict["mlp"] = mlp_track
            if request_logits is not None and len(request_logits) > 0:
                ret_dict["logits"] = layer_logits_track
            return txt[0], ret_dict