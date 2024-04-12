from codetrace.interp_utils import collect_hidden_states_at_tokens, insert_patch
from codetrace.utils import placeholder_to_std_fmt, STARCODER_FIM
import datasets
import torch
from nnsight import LanguageModel
import argparse
import os
import json 
from tqdm import tqdm

def main(args):
    """
    For each batch, collect activations of fim_middle token;
    add steering vector at layers if it exists.
    Save to outdir.
    
    NOTE: only does "fim_middle token"
    """
    os.makedirs(args.outdir, exist_ok=True)
    model = LanguageModel(args.model_name, device_map="cuda")
    list_ds = []
    for ds_name in args.local_result_datasets:
        ds = datasets.load_from_disk(ds_name)
        list_ds.append(ds)
    data = datasets.concatenate_datasets(list_ds)
    data = data.shuffle(seed=42)
    if args.max_size > -1:
        data = data.select(range(args.max_size))
    print(data)
    
    if args.steering_tensor !=  None:
        steering_tensor = torch.load(args.steering_tensor)
    else:
        steering_tensor = None
        
    batched = [data[i:i+args.batchsize] for i in range(args.start_at_index, len(data), args.batchsize)]
    for i,batch in tqdm(enumerate(batched), desc="Collecting activations...", total=len(batched)):
        if "<FILL>" in batch["fim_program"][0]:
            prompts = [placeholder_to_std_fmt(b, STARCODER_FIM) for b in batch["fim_program"]]
        else:
            prompts = batch["fim_program"] 
            
        if steering_tensor == None:
            hs = collect_hidden_states_at_tokens(model, prompts, "<fim_middle>")
        else:
            tr = insert_patch(model, prompts, steering_tensor, args.layers_to_patch, "<fim_middle>", collect_hidden_states=True)
            hs = tr._hidden_states
            hs = hs[:,:,-1,:]
            
        # save hidden states
        if args.start_at_index > 0:
            l = i*args.start_at_index
        torch.save(hs, f"{args.outdir}/activations_{i}.pt")
        correct_label = [bool(v) for v in batch["correct_steer"]]
        with open(f"{args.outdir}/labels_{i}.json", "w") as f:
            json.dump(correct_label, f)
        del(hs)
        
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True, type=str)
    parser.add_argument("--local-result-datasets", nargs="+", required=True, type=str)
    parser.add_argument("--outdir", required=True, type=str)
    parser.add_argument("--batchsize", type=int, default=8)
    parser.add_argument("--steering-tensor", type=str, default=None)
    parser.add_argument("--layers-to-patch", type=int, nargs="+", default=[])
    parser.add_argument("--max-size", type=int, default=-1)
    parser.add_argument("--start-at-index", type=int, default=0)
    args = parser.parse_args()
    main(args)