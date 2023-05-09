import torch
import transformers



def gen_prompt(loaded_model, loaded_tok, prompts, context="", eos_tok=None, sep_tok="\n"):
    
    #apply context
    prompts_with_context = [context + sep_tok + prompt for prompt in prompts]
    
    tokenized = loaded_tok(prompts_with_context, padding=True, return_tensors="pt").to(loaded_model.device)
    
    if eos_tok not None:
        eos_tok = loaded_tok.encode(eos_tok)
    
    outputs = loaded_model.generate(tokenized, max_len=max_len, forced_eos_token=eos_tok)
    
    return loaded_tok.decode(outputs)