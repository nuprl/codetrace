from argparse import ArgumentParser, Namespace
from collections import Counter
import datasets
from nnsight import LanguageModel
import json
import torch
from codetrace.interp_utils import arg_to_list,insert_patch
import os
from typing import List, Union
import sys
import pandas as pd
from tqdm import tqdm
import gzip
import re
from codetrace.utils import placeholder_to_std_fmt, STARCODER_FIM, unfim
from codetrace.type_inf_exp.scripts.steering import fit_test_split, _pretty_print, _get_steering_tensor

def steer(
    model,
    dataset : datasets.Dataset, #codegen dataset
    patch_tensor : torch.Tensor,
    layers_to_patch : Union[List[int],int],
    tokens_to_patch : Union[List[int],int],
    patch_mode : str,
    batch_size : int,
    max_out : int = 512,
    outfile : str = None
):
    """
    Need to steer with generation
    TODO: efficient stopping
    """
    prompts = dataset["renamed_prompt"]
    # stop_tokens = dataset["stop_tokens"]
    layers_to_patch, tokens_to_patch = arg_to_list(layers_to_patch), arg_to_list(tokens_to_patch)
    assert isinstance(tokens_to_patch[0],int), "For generation, tokens_to_patch must be int"
        
    # prepare batches
    prompt_batches = [prompts[i:i+batch_size] for i in range(0, len(prompts), batch_size)]
    results = []
    for b in tqdm(range(len(prompt_batches)), desc="Steer batch prompt", total=len(prompt_batches)):
        batch = prompt_batches[b]
        # make into fim
        batch = [placeholder_to_std_fmt(p+"<FILL>", STARCODER_FIM) for p in batch]

        for i in tqdm(range(max_out), desc="token"):
            out = insert_patch(model, batch, patch_tensor, layers_to_patch, tokens_to_patch, patch_mode, collect_hidden_states=False)
            logits = out.decode_logits(prompt_idx=list(range(len(batch))))
            generated = []
            for j in range(len(batch)):
                tok = logits[-1][j].tokens(model.tokenizer)[0]
                generated.append(tok)
                
            # print("\nGenerated:",[bytes(g, "utf-8") for g in generated])
            # add new tokens to batch
            batch = [k + g for k,g in zip(prompt_batches[b], generated)]
            # add fim_middle token
            batch = [placeholder_to_std_fmt(k+"<FILL>", STARCODER_FIM) for k in batch]
            
        batch = [unfim(k, STARCODER_FIM) for k in batch]
        for j in range(len(batch)):
            results.append({
                **dataset[b*batch_size + j],
                "generated" : batch[j]
            })
            
        if outfile is not None:
            # save results
            pd.DataFrame(results).to_csv(outfile)
            
    return datasets.Dataset.from_pandas(pd.DataFrame(results))

# apply stop tokens
def _apply_stop_tokens(prompt, completion, stop_tokens):
    def stop_at_stop_token(decoded_string, stop_tokens):
        """
        Produces the prefix of decoded_string that ends at the first occurrence of
        a stop_token.

        WARNING: the decoded_string *must not* include the prompt, which may have stop tokens
        itself.
        """
        min_stop_index = len(decoded_string)
        for stop_token in stop_tokens:
            stop_index = decoded_string.find(stop_token)
            if stop_index != -1 and stop_index < min_stop_index:
                min_stop_index = stop_index
        return decoded_string[:min_stop_index].strip()
    
    completion = completion.replace(prompt, "")
    return stop_at_stop_token(completion, stop_tokens + ["<|endoftext|>", 
                                                            "<fim_prefix>",
                                                            "<fim_suffix>", 
                                                            "<fim_middle>"])


def main():
    print("BEWARE! This script is not yet fully tested and may not work as expected")
    # ==========================================================================================
    # PART 0: setup
    # ==========================================================================================
    steering_args = sys.argv[1]
    with open(steering_args, "r") as f:
        args = json.load(f)
    args = Namespace(**args)

    # parent dir
    exp_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    args.outdir = f"{exp_dir}/exp_data/{args.outdir}"
    os.makedirs(args.outdir, exist_ok=True)
    
    ds = datasets.load_dataset(args.dataset, split="train")

    print(ds)
    
    model = LanguageModel(args.model, device_map="cuda")

    # ==========================================================================================
    # PART 2: averages (provided)
    # ==========================================================================================
    if os.path.exists(f"{args.outdir}/steering_tensor.pt"):
        print(f"...Loading steering tensor from {args.outdir}/steering_tensor.pt...")
        steering_tensor = torch.load(f"{args.outdir}/steering_tensor.pt")
    else:
        raise ValueError("Steering tensor not found--needed for codegen")
    print(f"Steering tensor shape: {steering_tensor.shape}")

    #==========================================================================================
    # Part 3: steered generations
    #==========================================================================================
    
    print(f"...Applying patch to incorrect prompts...")
        
    steering_ds = steer(model, 
                        ds,
                        steering_tensor,
                        args.layers_to_patch, 
                        args.tokens_to_patch, 
                        args.patch_mode,
                        args.batch_size,
                        args.max_out,
                        f"{args.outdir}/steering_results.csv")
    
    # save results
    steering_ds.save_to_disk(f"{args.outdir}/steering_results_ds_unfiltered")
    print(steering_ds)
    
    def _is_changed(old_results, new_results, tests):
        assert len(old_results) == len(new_results) == 1, "Only one completion"
        old_results = old_results[0]["program"]
        new_results = new_results[0]
        old_results = old_results.replace(tests, "")
        new_results = new_results.replace(tests, "")
        old_results_non_whitespace = re.sub(r"\s+", "", old_results)
        new_results_non_whitespace = re.sub(r"\s+", "", new_results)
        return not new_results_non_whitespace.startswith(old_results_non_whitespace)
    
    # save into MultiPLE completions format
    if "results" in steering_ds.column_names:
        steering_ds = steering_ds.rename_columns({"results":"old_results"})
        steering_ds = steering_ds.map(lambda x : {**x, "completions" : [x["generated"].strip()+"\n"+x["tests"]]})
        steering_ds = steering_ds.map(lambda x : {**x, 'changed_after_steering': _is_changed(x["old_results"], x["completions"], x["tests"]),
                                                'top_p': 0.95, # TODO: fix this?
                                                'max_tokens': args.max_out
                                                })
    else:
        steering_ds = steering_ds.map(lambda x : {**x, "completions" : [x["generated"].strip()+"\n"+x["tests"]]})
        steering_ds = steering_ds.map(lambda x : {**x, 'changed_after_steering': None,
                                                'top_p': 0.95, # TODO: fix this?
                                                'max_tokens': args.max_out
                                                })
    # count num changed
    print(f"Num changed after steering: {Counter(steering_ds['changed_after_steering'])}")
    # save results
    steering_ds.save_to_disk(f"{args.outdir}/steering_results_ds")
    
    if "results" in steering_ds.column_names:
        steering_ds = steering_ds.rename_columns({"results":"completions", "renamed_prompt":"prompt"}) 
    else:
        steering_ds = steering_ds.rename_columns({"renamed_prompt":"prompt"})
    steering_ds = steering_ds.map(lambda x : {**x, 'completions': [_apply_stop_tokens(x["prompt"], x["completions"][0], x["stop_tokens"])]})
    
    steering_ds.to_csv(f"{args.outdir}/debug.csv")
    # save as gzip of json
    json_list = steering_ds.to_list()
    dirout = f"{args.outdir}/steering_completions"
    os.makedirs(dirout, exist_ok=True)
    for ex in json_list:
        with gzip.open(f"{dirout}/{ex['name']}.json.gz", "wt") as f:
            json.dump(ex, f)
            
            
if __name__ == "__main__":
    main()