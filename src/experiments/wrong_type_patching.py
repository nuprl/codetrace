"""
Make a dataset of good/bad <FIM> prompts for a given model.
"""
import datasets
import glob
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils import *
from tqdm import tqdm
import difflib
import pandas as pd
import random
from nnsight import LanguageModel
import torch
from nnsight import util
from nnsight.tracing.Proxy import Proxy
 

def make_dataset(model, tokenizer) -> datasets.Dataset:
    """Make a dataset of good/bad <FIM> prompts for a given model."""
    # use ts dataset
    fim_programs = []
    for f in glob.glob("ts-benchmark/*/*.ts"):
        prog = open(f).read()
        variations = fim_prog_func(prog)
        fim_programs += [{"program":v[0],
                          "solution":v[1],
                          "filename":f,
                          "variation_idx":i} for i,v in enumerate(variations)]
        
    examples = []
    random.seed(42)
    random.shuffle(fim_programs)
    
    for dikt in tqdm(fim_programs[:300]):
        program = dikt["program"]
        solution = dikt["solution"]
        p = placeholder_to_std_fmt(program, STARCODER_FIM)
        tokens = tokenizer(p, return_tensors="pt").to("cuda")
        input_ids = tokens.input_ids
        output = model.generate(input_ids, do_sample=False, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)
        full_out_text = tokenizer.decode(output[0])
        # get text after STARCODER_FIM.token
        generated_text = full_out_text.split(STARCODER_FIM.token)[-1]
        # remove special tokens
        generated_text = generated_text.replace(tokenizer.eos_token, "").replace(tokenizer.bos_token, "")
        examples.append({"prompt": program, 
                             "solution": solution, 
                             "generated": generated_text, 
                             "success" : (solution.strip() == generated_text.strip()),
                             "filename" : dikt["filename"],
                             "variation_idx" : dikt["variation_idx"]})
    ds = datasets.Dataset.from_list(examples)
    return ds
    
    
        
def get_avg_activations(model: LanguageModel,
                           prompts: list[str],
                           layers: List[int],
                           token_idx: int,
                           fim : FimObj):
    """
    Get the average activation of a token in a layer for a list of prompts
    """
    for p in tqdm(prompts):
        prompt = placeholder_to_std_fmt(p, fim)
        
        with model.forward() as runner:
            target_layers = {i:[] for i in layers}
            # Clean run
            with runner.invoke(prompt) as invoker:
                
                for layer_idx in layers:
                    target_layers[layer_idx].append(model.transformer.h[layer_idx].output[0].t[token_idx].save())
    
    for layer, activations in target_layers.items():
        target_layers[layer] = torch.stack(activations).mean(dim=0)
               
    return target_layers

def get_fim_avg_activations(model: LanguageModel,
                           prompts: list[str],
                           layer_idx: int,
                           token_idx: int,
                           fim : FimObj):
    """
    Get the average activation of a token in a layer for a list of prompts
    """
    activations_middle = []
    activations_suffix = [] 
    activations_prefix = []
    for p in tqdm(prompts):
        prompt = placeholder_to_std_fmt(p, fim)
        
        with model.forward() as runner:

            # Clean run
            with runner.invoke(prompt) as invoker:
                
                suffix_id = model.tokenizer.encode(fim.suffix)[0]
                prefix_id = model.tokenizer.encode(fim.prefix)[0]
                middle_id = model.tokenizer.encode(fim.token)[0]

                tokens = invoker.input["input_ids"][0]
                    
                for idx, t in enumerate(tokens.numpy().tolist()):
                    if t == suffix_id:
                        patch_suffix_idx = idx
                    if t == prefix_id:
                        patch_prefix_idx = idx
                    if t == middle_id:
                        patch_middle_idx = idx
                
                middle_hs = model.transformer.h[layer_idx].output[0].t[patch_middle_idx]
                prefix_hs = model.transformer.h[layer_idx].output[0].t[patch_prefix_idx]
                suffix_hs = model.transformer.h[layer_idx].output[0].t[patch_suffix_idx]
                activations_middle.append(middle_hs.save())
                activations_prefix.append(prefix_hs.save())
                activations_suffix.append(suffix_hs.save())
                
                
    return {"middle":torch.stack(activations_middle).mean(dim=0), 
            "prefix":torch.stack(activations_prefix).mean(dim=0),
            "suffix":torch.stack(activations_suffix).mean(dim=0)}


def perform_patch(model: LanguageModel,
                  prompts: list[str],
                  solutions: list[str],
                  layer_idx: list[int],
                  token_idx: int,
                  patches: Dict[int, torch.Tensor],
                  fim : FimObj,
                  num_tokens: int = 5):
    
    def greedy_decode(hs : torch.Tensor):
        return model.lm_head(model.transformer.ln_f(hs))
    
    results = []
    model.tokenizer.padding_side = "left"
    for i,p in tqdm(enumerate(prompts)):
        prompt = placeholder_to_std_fmt(p, fim)
        new = ""
        for n in range(num_tokens):
            prompt += new
            with model.generate(max_new_tokens=1) as generator:
                
                suffix_id = model.tokenizer.encode(fim.suffix)[0]
                prefix_id = model.tokenizer.encode(fim.prefix)[0]
                middle_id = model.tokenizer.encode(fim.token)[0]
                
                with generator.invoke(prompt) as invoker:
                    tokens = invoker.input["input_ids"][0]
                    
                    for idx, t in enumerate(tokens.numpy().tolist()):
                        if t == suffix_id:
                            patch_suffix_idx = idx
                        if t == prefix_id:
                            patch_prefix_idx = idx
                        if t == middle_id:
                            patch_middle_idx = idx
                           
                    for l in layer_idx:
                        pass
                        # model.transformer.h[l].output[0].t[-1] += patch
                        # model.transformer.h[l].output[0].t[patch_middle_idx] = patches[l]
                        
                        
                        # model.transformer.h[layer_idx].output[0].t[patch_suffix_idx] += patch
                        # model.transformer.h[layer_idx].output[0].t[patch_prefix_idx] += patch
                        # hidden_states = model.transformer.h[-1].output[0].save()
                    invoker.next()
                    
                    # model.transformer.h[layer_idx].output[0].t[token_idx] += patch
                    hidden_states = model.transformer.h[-1].output[0].save()
                    # patched_logits = model.lm_head.output.save()
                    
                    # # get the prediction
                    # probs = patched_logits.softmax(dim=-1)
                    # # find argmax of -1, prob shape is [1,2,vocab_size]
                    # max_logit = probs[0, -1].argmax().save()
                    
            out = generator.output
            out = util.apply(out, lambda x: x.value, Proxy)
            new = model.tokenizer.decode(out.cpu().numpy().tolist()[0][-1])
            
            
        solution_idx = model.tokenizer.encode(solutions[i])[-1]
        results.append({"prompt":p, 
                            "generated":out, 
                            "solution":solutions[i],})
                
    return results


def perform_patch_full(model: LanguageModel,
                    prompts: list[str],
                    solutions: list[str],
                    layer_idx: list[int],
                    token_idx: int,
                    patch_mid: torch.Tensor,
                    patch_pre: torch.Tensor,
                    patch_suf: torch.Tensor,
                    fim : FimObj,
                    num_tokens: int = 5):
    
    results = []
    model.tokenizer.padding_side = "left"
    for i,p in tqdm(enumerate(prompts)):
        prompt = placeholder_to_std_fmt(p, fim)
        new = ""
        for n in range(num_tokens):
            prompt += new
            with model.generate(max_new_tokens=1) as generator:
                
                suffix_id = model.tokenizer.encode(fim.suffix)[0]
                prefix_id = model.tokenizer.encode(fim.prefix)[0]
                middle_id = model.tokenizer.encode(fim.token)[0]

                with generator.invoke(prompt) as invoker:
                    
                    tokens = invoker.input["input_ids"][0]
                    
                    for idx, t in enumerate(tokens.numpy().tolist()):
                        if t == suffix_id:
                            patch_suffix_idx = idx
                        if t == prefix_id:
                            patch_prefix_idx = idx
                        if t == middle_id:
                            patch_middle_idx = idx
                           
                    for l in layer_idx:
                        # pass
                        # model.transformer.h[l].output[0].t[-1] += patch
                        model.transformer.h[l].output[0].t[patch_middle_idx] = patch_mid
                        model.transformer.h[l].output[0].t[patch_suffix_idx] = patch_suf
                        model.transformer.h[l].output[0].t[patch_prefix_idx] = patch_pre
                        # model.transformer.h[layer_idx].output[0].t[patch_suffix_idx] += patch
                        # model.transformer.h[layer_idx].output[0].t[patch_prefix_idx] += patch
                        # hidden_states = model.transformer.h[-1].output[0].save()
                    invoker.next()
                    
                    # model.transformer.h[layer_idx].output[0].t[token_idx] += patch
                    hidden_states = model.transformer.h[-1].output[0].save()
                        
                    # patched_logits = model.lm_head.output.save()
                    
                    # # get the prediction
                    # probs = patched_logits.softmax(dim=-1)
                    # # find argmax of -1, prob shape is [1,2,vocab_size]
                    # max_logit = probs[0, -1].argmax().save()
                    
            out = generator.output
            out = util.apply(out, lambda x: x.value, Proxy)
            new = model.tokenizer.decode(out.cpu().numpy().tolist()[0][-1])
            
        solution_idx = model.tokenizer.encode(solutions[i])[-1]
        results.append({"prompt":p, 
                            "generated":out, 
                            "solution":solutions[i]})
                
    return results