import torch
from trace_utils import *
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import warnings
from typing import Union, List
from model_utils import untuple, extract_layer_formats
import numpy as np
import unicodedata
from pathlib import Path
import os
from logit_lens import LogitLens
import numpy
from collections import defaultdict
from baukit.nethook import get_module, set_requires_grad
from model_utils import *
'''

'''


class ModelLoader:
    def __init__(self, 
                 model_name_or_path, 
                 AUTH=True,
                 dtype = torch.float32, ## required by trace dict
                 trust_remote_code=True) -> None:
        
        self.model_name = model_name_or_path
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, 
                                                       use_auth_token=AUTH) 
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            output_attentions=True,
            use_auth_token=AUTH,
            low_cpu_mem_usage=True, ## loads with accelerate
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
            
        )
        self.model.eval().cuda()
         
        # post process
        self.extract_fields()
        
        set_requires_grad(False, self.model)
        
        ## set pad tokens
        self.tokenizer.clean_up_tokenization_spaces=False
        self.tokenizer.padding_side="left"
        self.tokenizer.pad_token = self.tokenizer.eos_token  
            

        
        
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
        
        self.fast_generation = (self.model_type not in ["galactica", "llama", "gpt2", "gpt_bigcode"])

        if(self.model_type is not None):
            self.layer_names = [self.layer_name_format.format(i) for i in range(self.model.config.n_layer)]
            self.mlp_module_names = [self.mlp_module_name_format.format(i) for i in range(self.model.config.n_layer)]
            self.attn_module_names = [self.attn_module_name_format.format(i) for i in range(self.model.config.n_layer)]
            self.tracable_modules =  self.mlp_module_names + self.attn_module_names + self.layer_names
            
            
            
    def trace_generate(
            self,
            prompts: Union[str, List[str]],
            top_k: int = 5,                 
            max_out_len: int = 20,          
            argmax_greedy = False, 
            debug = False,
            quiet=False,
            request_activations = None,
            request_logits = None,
        ):
        '''
        Trace with cache implementation if possible
        '''
        if max_out_len < len(max(prompts, key=len)):
            raise ValueError("Prompt length exceeds max_out_len")
        
        request_activations = [] if request_activations is None else request_activations
        request_logits = [] if request_logits is None else request_logits
        
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
            invalid_module = list(set(request_activations) - set(self.tracable_modules))
            assert(
                len(invalid_module) == 0
            ), f"modules {invalid_module} are not in the list of tracable modules"
            layer_logits_track = {k: None for k in request_logits}
        if(type(prompts) == str):
            prompts = [prompts]

        
        tokenized = self.tokenizer(prompts, padding=True, return_tensors="pt").to(self.model.device)
        
        input_ids, attention_mask = tokenized["input_ids"], tokenized["attention_mask"]
        # print(input_ids)
        batch_size = input_ids.size(0)

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

        # fast gen not supported bigcode
        if not self.fast_generation:
            use_cache = False
            warnings.warn(f"The model `{self.model_name}` of type `{self.model_type}` already implements or can't utilize `use_cache` for fast generation. Setting `use_cache = False`.")

        generated_tokens = [[] for _ in range(input_ids.size(0))]
        with torch.no_grad():
            # while not exceeding max output length
            while input_ids.size(1) < max_out_len: 

                ## traces curr inputs to prediction of next tok
                with TraceDict(
                    self.model, 
                    layers = request_activations+request_logits,
                    retain_input=True,
                ) as traces:
                    model_out = self.model(
                        input_ids=input_ids[:, cur_context],
                        attention_mask=attention_mask[:, cur_context],
                        past_key_values=past_key_values,
                        use_cache = use_cache,  
                    )
                # print(traces[request_activations[0]].__dir__())

                if(len(request_activations) > 0):
                    assert self.layername(0, "embed") not in request_activations, "Embedding layer is not supported"
                    if(input_ids.size(1) == max_out_len - 1):
                        for module in request_activations:
                            # print("traces shape:", untuple(traces[module].output).shape)
                            if(activation_track[module] is None):
                                activation_track[module] = untuple(traces[module].block_output).cpu().numpy()
                            if(attn_track[module] is None):
                                attn_track[module] = untuple(traces[module].attn_output).cpu().numpy()
                            if(mlp_track[module] is None):
                                mlp_track[module] = untuple(traces[module].mlp_output).cpu().numpy()
                if(len(request_logits) > 0):
                    assert self.layername(0, "embed") not in request_logits, "Embedding layer is not supported"
                    if(input_ids.size(1) == max_out_len - 1):
                        for module in request_logits:
                            # print("traces shape:", untuple(traces[module].output).shape)
                            if(layer_logits_track[module] is None):
                                layer_logits_track[module] = untuple(traces[module].block_output).cpu().numpy()
                                
                final_logits, past_key_values = model_out.logits, model_out.past_key_values

                softmax_out = torch.nn.functional.softmax(final_logits[:, -1, :], dim=1)

                # Top-k sampling
                tk = torch.topk(softmax_out, top_k, dim=1).indices
                softmax_out_top_k = torch.gather(softmax_out, 1, tk)
                softmax_out_top_k = softmax_out_top_k / softmax_out_top_k.sum(1)[:, None]

                if(argmax_greedy == False):
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

                if(debug == True):
                    for i in range(input_ids.size(0)):
                        formatted = [(g["token"], np.round(g["p"], 4)) for g in generated_tokens[i][-1]]
                        print(f"prompt <{i}> ==> {formatted}")
                    if(input_ids.size(0) > 1):
                        print()

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
                    input_ids[i][new_idx] = new_toks[i]
                    attention_mask[i][new_idx] = 1
                    cur_context = slice(0, new_idx.item()+1)

            ## End gen loop:

            cur_context = slice(0, cur_context.stop + 1)

            # clear up GPU
            # del(traces)
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
                ret_dict["activations"] = {
                    "block" : activation_track,
                    "attn" : attn_track,
                    "mlp" : mlp_track,
                }
            if request_logits is not None and len(request_logits) > 0:
                ret_dict["logits"] = self.get_logits(layer_logits_track, top_k=top_k)
            return txt, ret_dict

    def get_logits(
            self,
            activations,
            top_k: int = 5,
    ):

        llens_gen = LogitLens(
            self.model,
            self.tokenizer,
            activations = activations,
            top_k=top_k,
        )
        return llens_gen()
        
            
    # def get_logits(
    #         self,
    #         prompts: Union[str, List[str]],
    #         top_k: int = 5,                 
    #     ):
    #         if(type(prompts) == str):
    #             prompts = [prompts]
                
    #         layer_module_tmp = "transformer.h.{}"
    #         ln_f_module = "transformer.ln_f"
    #         lm_head_module = "lm_head"

    #         llens_gen = LogitLens(
    #             self.model,
    #             self.tokenizer,
    #             layer_module_tmp,
    #             ln_f_module,
    #             lm_head_module,
    #             disabled=False,
    #         )
    #         inp_prompt = self.tokenizer(prompts, padding=True, return_tensors="pt").to(
    #             self.model.device
    #         )
    #         with llens_gen:
    #             self.model(**inp_prompt)
    #         print("\n--- Argument Model Logit Lens ---")
    #         llens_gen.pprint(k=top_k)
        
        
    def patch_hidden_states(
        self,
        prompts,
        state_to_state, # (layer_from, layer_to, start_tok, end_tok)
        answers_t=None,  # Answer probabilities to collect
        noise=0.1,  # Level of noise to add
        uniform_noise=False,
        replace=False,  # True to replace with instead of add noise
        trace_layers=None,  # List of traced outputs to return
    ):
       
        toks = self.tokenizer(prompts, padding=True, return_tensors="pt").to(self.model.device)
        inp = toks["input_ids"]
        
        # with torch.no_grad():
        #     answers_t, base_score = [d[0] for d in predict_from_input(self.model, inp)]
        # attn_mask = toks["attention_mask"]
        # if answers_t is None:
        #     # all probs 
        #     answers_t = torch.ones(inp["input_ids"].shape[0], dtype=torch.long).to(self.model.device)
            
        rs = numpy.random.RandomState(1)  # For reproducibility, use pseudorandom noise
        if uniform_noise:
            prng = lambda *shape: rs.uniform(-1, 1, shape)
        else:
            prng = lambda *shape: rs.randn(*shape)

        patch_spec = defaultdict(list)
        for h, l in heads_to_patch:
            patch_spec[l].append(h)

        embed_layername = self.layername( 0, "embed")
        
        # keep embed layer uncorrupted
        assert patch_spec[embed_layername] != []
        
        def untuple(x):
            return x[0] if isinstance(x, tuple) else x

        # Define the model-patching rule.
        if isinstance(noise, float):
            noise_fn = lambda x: noise * x
        else:
            noise_fn = noise

        h_dim = int(self.model.config.n_embd / self.model.config.n_head)
        
        layer_to_copy = self.layername(6)
        saved_x = []
        
        def patch_rep(x, layer): # x is the output of the layer
            if layer == self.layername(6):
                saved_x.append(x)
                return x
            elif layer == self.layername(39):
                return saved_x[0]
            
            for h in range(48):
                if h not in patch_spec[layer]: 
                    noise_data = noise_fn(
                        torch.from_numpy(prng(x[0].shape[0], x[0].shape[1], h_dim))
                    ).to(x[0].device)
                    # print(noise_data.shape)
                    if replace:
                        x[0][:,:,h*h_dim:(h+1)*h_dim] = noise_data # 0 is tuple
                    else:
                        x[0][:,:,h*h_dim:(h+1)*h_dim] += noise_data
                        
            return x
            

        # With the patching rules defined, run the patched model in inference.
        additional_layers = [] if trace_layers is None else trace_layers
        with torch.no_grad(), TraceDict(
            self.model,
            [embed_layername] + [self.layername(i) for i in range(1, 40)],
            edit_output=patch_rep,
        ) as td:
            outputs_exp = self.model(inp)

        # We report softmax probabilities for the answers_t token predictions of interest.
        probs = torch.softmax(outputs_exp.logits, dim=-1).mean(dim=0)[-1] # last token for each batch
        

        # If tracing all layers, collect all activations together to return.
        if trace_layers is not None:
            all_traced = torch.stack(
                [untuple(td[layer].output).detach().cpu() for layer in trace_layers], dim=2
            )
            return probs, all_traced

        return probs
    
    def patch_heads(
        self,
        prompts,
        heads_to_patch # (head_index, layername)
    ):
        pass
    
    
    def layername(self, num, kind=None):
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
      
    def search_causal_heads(self, prompt, layers = range(20,31), replace=False, noise=0.9):
        heads_to_patch = []
        for l in layers:
            layername = self.layername(l)
            heads_to_patch += [(i, layername) for i in range(48)]
            
        probs = self.trace_with_patch(prompt, heads_to_patch=heads_to_patch, 
                                replace=replace, noise = noise)
        top_completion = self.tokenizer.decode(probs.argmax(dim=0))
        # print(top_completion, heads_to_patch)
        try:
            tc = int(top_completion)
        except:
            return []
            
        return heads_to_patch