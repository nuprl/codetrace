import torch
from baukit import nethook
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map
import warnings
import time
from typing import Union, List
from model_utils import untuple, extract_layer_formats
import numpy as np
import json
import unicodedata
'''

'''


class ModelLoader:
    def __init__(self, 
                 MODEL_NAME_OR_PATH, 
                 MODEL_NAME = None,
                 dtype = torch.float32,
                 trust_remote_code=True,
                 preloaded_model=None) -> None:
        
        if MODEL_NAME is None:
            self.MODEL_NAME = MODEL_NAME_OR_PATH
        else:
            ## for downloaded converted weights
            self.MODEL_NAME = MODEL_NAME
            
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH) 
        
        if preloaded_model is None:
            start_time = time.process_time_ns()
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME_OR_PATH, 
                output_attentions=True,
                low_cpu_mem_usage=True, ## load with accelerate
                torch_dtype=dtype,
                trust_remote_code=trust_remote_code
            )
            print(f"Load time: {time.process_time_ns()-start_time} ns") 
            self.model.eval().cuda()
        else:
            self.model = preloaded_model

        self.extract_fields()
        
        nethook.set_requires_grad(False, self.model)


        for n, p in self.model.named_parameters():
            print(n, p.shape, p.device)

        ## set pad tokens
        if(self.model_type in ["gpt2", "gpt_neox", "llama"]):
            self.tokenizer.pad_token = self.tokenizer.eos_token            
        elif(self.model_type in [ "galactica"]):
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'}) 
            

        
        
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
        
        self.fast_generation = (self.model_type not in ["galactica", "llama", "gpt2"])

        if(self.model_type is not None):
            self.layer_names = [self.layer_name_format.format(i) for i in range(self.model.config.n_layer)]
            self.mlp_module_names = [self.mlp_module_name_format.format(i) for i in range(self.model.config.n_layer)]
            self.attn_module_names = [self.attn_module_name_format.format(i) for i in range(self.model.config.n_layer)]
            self.tracable_modules =  self.mlp_module_names + self.attn_module_names + self.layer_names
            
            
            
    def generate(
            self,
            prompts: Union[str, List[str]], # TODO: technically it will accept a list of prompts, but due to some unresolved bugs generate doesn't work well with prompts of different sizes.
            top_k: int = 5,                 
            max_out_len: int = 20,          
            argmax_greedy = False,          # if top_k=1 it is by defaults generate greedy. Otherwise, it will report `top_k` predictions but pick the top one
            debug = False,
            use_cache = True,

            request_activations = None
        ):
        '''
        Trace with cache implementation if possible
        '''

        request_activations = [] if request_activations is None else request_activations
        if(len(request_activations) > 0):
            invalid_module = list(set(request_activations) - set(self.tracable_modules))
            assert(
                len(invalid_module) == 0
            ), f"modules {invalid_module} are not in the list of tracable modules"
            activation_track = {k: None for k in request_activations}

        if(type(prompts) == str):
            prompts = [prompts]

        tokenized = self.tokenizer(prompts, padding=True, return_tensors="pt").to(self.model.device)

        input_ids, attention_mask = tokenized["input_ids"], tokenized["attention_mask"]

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
            warnings.warn(f"The model `{self.MODEL_NAME}` of type `{self.model_type}` already implements or can't utilize `use_cache` for fast generation. Setting `use_cache = False`.")

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
                    if(use_cache == True):
                        for module in request_activations:
                            # print("traces shape:", untuple(traces[module].output).shape)
                            if(activation_track[module] is None):
                                activation_track[module] = untuple(traces[module].output).cpu().numpy()
                            else:
                                activation_track[module] = np.concatenate(
                                    (activation_track[module], untuple(traces[module].output).cpu().numpy()),
                                    axis = 1
                                )
                    elif(input_ids.size(1) == max_out_len - 1):
                        for module in request_activations:
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

            if(use_cache == False):
                cur_context = slice(0, cur_context.stop + 1)
            else:
                cur_context = slice(cur_context.stop, cur_context.stop + 1)

            # clear up the precious GPU memory as soon as the inference is done
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
                ret_dict["activations"] = activation_track

            return txt, ret_dict

