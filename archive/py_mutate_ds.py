import datasets
import argparse
import pandas as pd
from multiprocessing import cpu_count
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm
from codetrace.fast_utils import make_batches, batched_apply
from codetrace.py_mutator import map_random_mutations
import os
from codetrace import py_mutator 
from codetrace.parsing_utils import get_model_fim
from codetrace.utils import load_dataset, save_dataset, num_available_devices, get_vllm_config
from pathlib import Path

def filter_incorrect(
    ds: datasets.Dataset,
    llm: LLM,
    new_ds_name:str,
    batch_size:int = 10000
) -> datasets.Dataset:
    """
    Filter out examples where the model's prediction is incorrect. Truncate generation and
    solution at 1 token
    """
    model_fim = get_model_fim(get_vllm_config(llm).name_or_path)
    params = SamplingParams(temperature=0, max_tokens=1)
    ds = ds.map(lambda x: {
            "prompt" : model_fim.placeholder_to_fim(x["mutated_program"]),
            "solution": x["fim_type"]
        }, desc="Prepping prompts")
    
    # batch generations so we can save them early
    new_ds = []
    for batch_idx,i in tqdm(
        enumerate(range(0, len(ds), batch_size)), 
        desc="Filtering out incorrect mutations", 
        total=len(ds) // batch_size
    ):
        use_tqdm = (batch_idx == 0)
        generations = llm.generate(ds["prompt"][i:i+batch_size], params, use_tqdm=use_tqdm)

        for j,output in enumerate(generations):
            generated_text = output.outputs[0].text.strip()
            if generated_text != ds[i+j]["solution"]:
                new_ds.append({**ds[i+j],"mutated_generated_text": generated_text})
                
        # save every
        print(f"Len new_ds: {len(new_ds)}")
        if len(new_ds) > 0:
            new_ds_hf = datasets.Dataset.from_pandas(pd.DataFrame(new_ds))
            save_dataset(new_ds_hf, new_ds_name, private=True)
    
    new_ds = datasets.Dataset.from_pandas(pd.DataFrame(new_ds))
    save_dataset(new_ds_hf, new_ds_name, private=True)
    new_ds = new_ds.remove_columns(["prompt", "solution"])
    return new_ds

def py_preprocess(iterable, correct_bool = True):
    """
    Preprocess the dataset
    - Take only correct examples
    """
    def _condition(x):
        return x["correct"] == correct_bool 
    
    if isinstance(iterable, datasets.Dataset):
        return iterable.filter(_condition, desc="Preprocess")
    else:
        return filter(_condition, iterable)

def _mutate_batch(batch, mutations, correct_bool = True):
    post = py_preprocess(batch, correct_bool=correct_bool)
    return map_random_mutations(post, mutations)

def main(
    model: LLM,
    ds: datasets.Dataset,
    new_ds_name: str,
    batch_size:int
):
    batches = make_batches(ds, cpu_count())
    results = batched_apply(batches, cpu_count(), _mutate_batch, mutations=mutations, correct_bool=args.correct_bool)
    ds = datasets.Dataset.from_list(results)
    save_dataset(ds, Path(new_ds_name + "_unfiltered"))
    
    ds = filter_incorrect(ds, model, new_ds_name, batch_size=batch_size)
    print(ds)
    save_dataset(ds, Path(new_ds_name))

# TODO remove comments or change pyright settings to ignore commenst (surprisingly, it doesnt)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--completions-ds", type=str, required=True)
    parser.add_argument("--mutated-ds", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--mutations", type=str, required=True, nargs="+", choices=["type","vars","delete"])
    
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--max-size", type=int, default=-1)
    parser.add_argument("--correct-bool", type=bool, default=True)
    parser.add_argument("--seed", type=int, default=None)
    
    args = parser.parse_args()
    datasets.disable_caching()
    print("Gpu:", os.environ["CUDA_VISIBLE_DEVICES"])

    ds = load_dataset(args.completions_ds, split=args.split)
    if args.max_size > -1:
        ds = ds.shuffle(seed=args.seed).select(range(args.max_size))
    mutations = [getattr(py_mutator, m) for m in args.mutations]

    tps = num_available_devices()
    print(f"Serving VLLM across {tps} GPUs.")
    llm = LLM(args.model, tensor_parallel_size=tps, tokenizer=args.tokenizer, dtype="bfloat16")
    batchsize = 1000*tps # still want to save some intermediate completions
     
    main(llm, ds, args.mutated_ds, batchsize)
