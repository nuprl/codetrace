from argparse import ArgumentParser, Namespace
from collections import Counter
from nnsight import LanguageModel
import json
import pandas as pd
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
):
    """
    Need to steer with generation
    """
    prompts = dataset["renamed_prompt"]
    layers_to_patch, tokens_to_patch = arg_to_list(layers_to_patch), arg_to_list(tokens_to_patch)
    assert isinstance(tokens_to_patch[0],int), "For generation, tokens_to_patch must be int"

    # prepare batches
    prompt_batches = [prompts[i:i+batch_size] for i in range(0, len(prompts), batch_size)]
    results = []
    for b, batch in enumerate(prompt_batches):
        for i in range(max_out):
            out = insert_patch(model, batch, patch_tensor, layers_to_patch, tokens_to_patch, patch_mode, collect_hidden_states=False)
            logits = out.decode_logits(prompt_idx=list(range(len(batch))),top_p=0.95)
            generated = []
            for j in range(len(batch)):
                tok = logits[-1][j].tokens(model.tokenizer)
                tok = tok[0].strip()
                generated.append(tok)
            # add new tokens to batch
            batch = [b + g for b,g in zip(batch, generated)]
            
        for j in range(len(batch)):
            results.append({
                "generated" : generated[j],
                **dataset[b*batch_size + j]
            })
            
    return datasets.Dataset.from_pandas(pd.DataFrame(results))

def main():
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

    #==========================================================================================
    # Part 3: steered generations
    #==========================================================================================
    
    print(f"...Applying patch to incorrect prompts...")

    steering_ds = steer(model, ds, steering_tensor, args.layers_to_patch, args.tokens_to_patch, args.patch_mode, args.batch_size, args.max_out)
    
    # save results
    steering_ds.save_to_disk(f"{args.outdir}/steering_results_ds")
    
    # save into MultiPLE completions format
    steering_ds = steering_ds.rename_columns({"results":"old_results"})
    steering_ds = steering_ds.map(lambda x : {"completions" : [x["generated"]], 
                                              "temperature": 0.0,
                                              'language': 'py',
                                              'top_p': 0.95,
                                              'max_tokens': args.max_out,
                                              **x
                                              })
    # save as gzip of json
    json_list = steering_ds.to_list()
    dirout = f"{out_dir}/steering_completions"
    os.makedirs(dirout, exist_ok=True)
    for ex in json_list:
        with gzip.open(f"{dirout}/{ex['name']}.json.gz", "wt") as f:
            json.dump(ex, f)
            
if __name__ == "__main__":
    main()