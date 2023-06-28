import torch
import nethook
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
            low_cpu_mem_usage=True, ## load with accelerate
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
            
        )
        self.model.eval().cuda()
         
        # post process
        self.extract_fields()
        
        nethook.set_requires_grad(False, self.model)

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
            use_cache = False,
            quiet=False,
            request_activations = None
        ):
        '''
        Trace with cache implementation if possible
        '''
        assert(max_out_len > len(max(prompts, key=len))), "Prompt length exceeds max_out_len"
        request_activations = [] if request_activations is None else request_activations
        # print(self.tracable_modules)
        if(len(request_activations) > 0):
            invalid_module = list(set(request_activations) - set(self.tracable_modules))
            assert(
                len(invalid_module) == 0
            ), f"modules {invalid_module} are not in the list of tracable modules"
            activation_track = {k: None for k in request_activations}

        if(type(prompts) == str):
            prompts = [prompts]

        self.tokenizer.clean_up_tokenization_spaces = False
        tokenized = self.tokenizer(prompts, padding=True, return_tensors="pt").to(self.model.device)
        
        input_ids, attention_mask = tokenized["input_ids"], tokenized["attention_mask"]
        # print(input_ids)
        batch_size = input_ids.size(0)

        ## curr tok dict
        report_input_tokenized = []
        for b in range(batch_size):
            curr_inp_tok = []
            for t, a in zip(input_ids[b], attention_mask[b]):
                if(a == 0):
                    break
                curr_inp_tok.append((self.tokenizer.decode(t), t.item()))
            report_input_tokenized.append((curr_inp_tok))
        ret_dict = {"input_tokenized": report_input_tokenized}


        # Setup storage of fast generation with attention caches.
        # `cur_context` is used to define the range of inputs that are not yet
        # stored in `past_key_values`. At each step, we are generating the
        # next token for the index at `cur_context.stop + 1`.
        past_key_values, cur_context = None, slice(0, attention_mask.sum(1).min().item())  # init size of context

        if not self.fast_generation:
            use_cache = False
            warnings.warn(f"The model `{self.model_name}` of type `{self.model_type}` already implements or can't utilize `use_cache` for fast generation. Setting `use_cache = False`.")

        generated_tokens = [[] for _ in range(input_ids.size(0))]
        with torch.no_grad():
            while input_ids.size(1) < max_out_len:  # while not exceeding max output length


                ## traces curr inputs to prediction of next tok
                with nethook.TraceDict(
                    self.model, 
                    layers = request_activations,
                ) as traces:
                    model_out = self.model(
                        input_ids=input_ids[:, cur_context],
                        attention_mask=attention_mask[:, cur_context],
                        past_key_values=past_key_values,
                        use_cache = use_cache,
                    )



                if(len(request_activations) > 0):
                    if(input_ids.size(1) == max_out_len - 1):
                        for module in request_activations:
                            # print("traces shape:", untuple(traces[module].output).shape)
                            if(activation_track[module] is None):
                                activation_track[module] = untuple(traces[module].output).cpu().numpy()


                logits, past_key_values = model_out.logits, model_out.past_key_values

                softmax_out = torch.nn.functional.softmax(logits[:, -1, :], dim=1)

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

            # clear up the precious GPU memory as soon as the inference is done
            # del(traces)
            # del(model_out)
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
                ret_dict["activations"] = activation_track

            return txt, ret_dict


    def get_logits(
            self,
            prompts: Union[str, List[str]],
            top_k: int = 5,                 
        ):
            if(type(prompts) == str):
                prompts = [prompts]
                
            layer_module_tmp = "transformer.h.{}"
            ln_f_module = "transformer.ln_f"
            lm_head_module = "lm_head"

                    
            
            llens_gen = LogitLens(
                self.model,
                self.tokenizer,
                layer_module_tmp,
                ln_f_module,
                lm_head_module,
                disabled=False,
            )
            inp_prompt = self.tokenizer(prompts, padding=True, return_tensors="pt").to(
                self.model.device
            )
            with llens_gen:
                self.model(**inp_prompt)
            print("\n--- Argument Model Logit Lens ---")
            llens_gen.pprint(k=top_k)
        
    # def make_inputs(self, prompts, device="cuda"):
    #     tokenizer = self.tokenizer
    #     token_lists = [tokenizer.encode(p) for p in prompts]
    #     maxlen = max(len(t) for t in token_lists)
    #     if "[PAD]" in tokenizer.all_special_tokens:
    #         pad_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index("[PAD]")]
    #     else:
    #         pad_id = 0
    #     input_ids = [[pad_id] * (maxlen - len(t)) + t for t in token_lists]
    #     # position_ids = [[0] * (maxlen - len(t)) + list(range(len(t))) for t in token_lists]
    #     attention_mask = [[0] * (maxlen - len(t)) + [1] * len(t) for t in token_lists]
    #     return dict(
    #         input_ids=torch.tensor(input_ids).to(device),
    #         #    position_ids=torch.tensor(position_ids).to(device),
    #         attention_mask=torch.tensor(attention_mask).to(device),
    # )
            
    def trace_with_patch(
        self,  # The model
        prompts,  # A set of input prompts
        heads_to_patch,  # A list of (head_index, layername) triples to restore
        answers_t=None,  # Answer probabilities to collect
        noise=0.1,  # Level of noise to add
        uniform_noise=False,
        replace=False,  # True to replace with instead of add noise
        trace_layers=None,  # List of traced outputs to return
    ):
       
        toks = self.tokenizer(prompts, padding=True, return_tensors="pt").to(self.model.device)
        # inp = self.make_inputs(prompts, device=self.model.device)["input_ids"]
        inp = toks["input_ids"]
        print(inp)
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
        
        def untuple(x):
            return x[0] if isinstance(x, tuple) else x

        # Define the model-patching rule.
        if isinstance(noise, float):
            noise_fn = lambda x: noise * x
        else:
            noise_fn = noise

        def patch_rep(x, layer): # x is the output of the layer
            if layer in patch_spec:
                # erase head activations
                for h in patch_spec[layer]:
                    pass
            print("OUTPUT", untuple(x), untuple(x).shape, layer)      
            # if layer == embed_layername:
            #     # If requested, we corrupt a range of token embeddings on batch items x[1:]
            #     if tokens_to_mix is not None:
            #         b, e = tokens_to_mix
            #         noise_data = noise_fn(
            #             torch.from_numpy(prng(x.shape[0] - 1, e - b, x.shape[2]))
            #         ).to(x.device)
            #         if replace:
            #             x[1:, b:e] = noise_data
            #         else:
            #             x[1:, b:e] += noise_data
            #     return x
            # if layer not in patch_spec:
            #     return x
            # # If this layer is in the patch_spec, restore the uncorrupted hidden state
            # # for selected tokens.
            # h = untuple(x)
            # for t in patch_spec[layer]:
            #     h[1:, t] = h[0, t]
            return x

        # With the patching rules defined, run the patched model in inference.
        additional_layers = [] if trace_layers is None else trace_layers
        with torch.no_grad(), nethook.TraceDict(
            self.model,
            [embed_layername] + list(patch_spec.keys()) + additional_layers,
            edit_output=patch_rep,
        ) as td:
            outputs_exp = self.model(inp)

        # We report softmax probabilities for the answers_t token predictions of interest.
        probs = torch.softmax(outputs_exp.logits, dim=-1).mean(dim=0)[-1] # last token
        

        # If tracing all layers, collect all activations together to return.
        if trace_layers is not None:
            all_traced = torch.stack(
                [untuple(td[layer].output).detach().cpu() for layer in trace_layers], dim=2
            )
            return probs, all_traced

        return probs
    

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
      