"""
Most basic experiment:
clean prompt - corrupted prompt pairs
Can we recover the clean prompt from the corrupted prompt?

Idea:
clean prompt is predict type X FIM - corrupted prompt is predict type Y FIM
where X != Y

We can assume starcoder will guess all primitive types as we have seen in tests
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils import *
import glob
import datasets
import pandas as pd
import torch
from tqdm import tqdm
from nnsight import LanguageModel

# TypeScript types
basic_types = [
    "string",
    "number",
    "boolean",
]


def collect_stenotype_prompts():
    """
    Collect snippets of ts programs from stenotype-interp-dataset
    containing basic type annotations.
    
    - break at function definitions
    """
    ds = datasets.load_dataset("franlucc/stenotype-eval-dataset", split="train")
    new_ds = []
    fim_placeholder = "<FILL>"
    for i, ex in enumerate(ds):
        prog = ex["content"]
        tree = TS_PARSER.parse(bytes(prog, "utf8"))
        query = TS_LANGUAGE.query(
            """
    ((type_annotation) @name)
        """
        )
        
        captures = query.captures(tree.root_node)
        for i,c in enumerate(captures):
            new_entry = ex.copy()
            text = c[0].text.decode("utf-8").replace(":", "").strip()
            if any([t == text for t in basic_types]):
                s = replace_between_points(prog, c[0].start_point, c[0].end_point, fim_placeholder)
                new_entry["fim_program"] = s
                new_entry["fim_type"] = text
                new_entry["fim_placeholder"] = fim_placeholder
                new_entry["fim_type_start"] = c[0].start_point
                new_entry["fim_type_end"] = c[0].end_point
                new_entry["fim_variation"] = i
                new_entry["fim_hexsha"] = ex["hexsha"]+"_"+str(i)
                new_ds.append(new_entry)
                
    new_ds = datasets.Dataset.from_pandas(pd.DataFrame(new_ds))
    return new_ds


def normalize_tokcount(ds: datasets.Dataset, 
                       fim_tokens : FimObj,
                       tok_count : int,
                       tokenizer):
    """
    Cut out start of prefix and end of suffix until tokcount is reached
    """
    result = []
    for ex in tqdm(ds):
        fim_prog = ex["fim_program"]
        fim_prog = fim_prog.replace(ex["fim_placeholder"], fim_tokens.token)
        tokenized = tokenizer(fim_prog)
        fim_id = tokenizer.encode(fim_tokens.token)[0]
        # find idx of fim_id
        idx = [i for i, x in enumerate(tokenized.input_ids) if x == fim_id][0]
        tokenized.input_ids = tokenized.input_ids[idx-tok_count//2:idx+tok_count//2]
        # convert back to string
        truncated = "".join([tokenizer.decode(t) for t in tokenized.input_ids])
        # back into fill format
        truncated = truncated.replace(fim_tokens.token, ex["fim_placeholder"])
        
        result.append({"fim_program": truncated, 
                       "fim_type": ex["fim_type"], 
                       "fim_placeholder": ex["fim_placeholder"], 
                       "fim_variation": ex["fim_variation"], 
                       "fim_hexsha": ex["fim_hexsha"]})
    return result



def patch_fim_tokens(model : LanguageModel,
                     clean_prompt : str, 
                     corrupted_prompt : str,
                     fim_tokens : FimObj,
                     correct_index : int,
                     incorrect_index : int):
    """
    Patch fim_tokens from corrupted prompt with the ones in clean_prompt
    """
    # Enter nnsight tracing context
    with model.forward() as runner:

        # Clean run
        with runner.invoke(clean_prompt) as invoker:
            # get token idx of fim_tokens in clean_prompt and corrupted_prompt
            suffix_id = model.tokenizer.encode(fim_tokens.suffix)[0]
            prefix_id = model.tokenizer.encode(fim_tokens.prefix)[0]
            middle_id = model.tokenizer.encode(fim_tokens.token)[0]
            
            clean_tokens = invoker.input["input_ids"][0]

            for i, t in enumerate(clean_tokens.numpy().tolist()):
                if t == suffix_id:
                    clean_suffix_idx = i
                if t == prefix_id:
                    clean_prefix_idx = i
                if t == middle_id:
                    clean_middle_idx = i
            
            # Get hidden states of all layers in the network.
            # We index the output at 0 because it's a tuple where the first index is the hidden state.
            # No need to call .save() as we don't need the values after the run, just within the experiment run.
            clean_hs = [
                model.transformer.h[layer_idx].output[0]
                for layer_idx in range(len(model.transformer.h))
            ]

            # Get logits from the lm_head.
            clean_logits = model.lm_head.output

            # Calculate the difference between the correct answer and incorrect answer for the clean run and save it.
            clean_logit_diff = (
                clean_logits[0, -1, correct_index] - clean_logits[0, -1, incorrect_index]
            ).save()

        # Corrupted run
        with runner.invoke(corrupted_prompt) as invoker:
            corrupted_logits = model.lm_head.output
            
            # Get token idx of fim_tokens in corrupted_prompt
            corrupted_tokens = invoker.input["input_ids"][0]
            for i, t in enumerate(corrupted_tokens):
                if t == suffix_id:
                    corrupted_suffix_idx = i
                if t == prefix_id:
                    corrupted_prefix_idx = i
                if t == middle_id:
                    corrupted_middle_idx = i

            # Calculate the difference between the correct answer and incorrect answer for the corrupted run and save it.
            corrupted_logit_diff = (
                corrupted_logits[0, -1, correct_index]
                - corrupted_logits[0, -1, incorrect_index]
            ).save()
        
        print("Patching...")
        
        # Iterate through all the layers
        
        patching_results = []
        patched_predictions = []
        for layer_idx in tqdm(range(8,15)):
            layer_patching_results = []
                
            with runner.invoke(corrupted_prompt) as invoker:
                
                for token_idx in [corrupted_middle_idx]:
                    if token_idx == corrupted_prefix_idx:
                        clean_patch = clean_prefix_idx
                    elif token_idx == corrupted_suffix_idx:
                        clean_patch = clean_suffix_idx
                    elif token_idx == corrupted_middle_idx:
                        clean_patch = clean_middle_idx
                    else:
                        raise Exception("Token idx not found")
                    
                    clean_patch = clean_middle_idx
                    # Apply the patch from the clean hidden states to the corrupted hidden states.
                    model.transformer.h[layer_idx].output[0].t[token_idx] = clean_hs[
                        layer_idx
                    ].t[clean_patch]

                patched_logits = model.lm_head.output.save()
                
                patched_logit_diff = (
                    patched_logits[0, -1, correct_index]
                    - patched_logits[0, -1, incorrect_index]
                )

                # Calculate the improvement in the correct token after patching.
                patched_result = (patched_logit_diff - corrupted_logit_diff) / (
                    clean_logit_diff - corrupted_logit_diff
                )
                
                # get the prediction
                probs = patched_logits.softmax(dim=-1)
                # find argmax of -1, prob shape is [1,2,vocab_size]
                max_logit = probs[0, -1].argmax()

                # collect probabilities of idx 800 2171 4398
                p = [probs[0, -1, 800].save(), probs[0, -1, 2171].save(), probs[0, -1, 4398].save()]
                
            
                patched_predictions.append(max_logit.save())
            
                patching_results.append(p)
            
    return patching_results, patched_predictions



if __name__ == "__main__":
    ds = collect_stenotype_prompts()
    ds.push_to_hub("franlucc/type_patching_v0")
    ds = datasets.load_dataset("franlucc/type_patching_v0", split="train")
    from transformers import AutoTokenizer
    
    starcoder_fim = FimObj("<fim_prefix>", "<fim_suffix>","<fim_middle>", "<FILL>")
    res = normalize_tokcount(ds, 
                             starcoder_fim, 
                             50, 
                             AutoTokenizer.from_pretrained("bigcode/starcoderbase-1b"))
    ds_norm = datasets.Dataset.from_pandas(pd.DataFrame(res))
    ds_norm.push_to_hub("franlucc/type_patching_v0_50tok")
                    
    
    