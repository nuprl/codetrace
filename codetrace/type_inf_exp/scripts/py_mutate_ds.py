import datasets
import argparse
import pandas as pd
from multiprocessing import cpu_count
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm
from codetrace.fast_utils import make_batches, batched_apply
from codetrace.type_inf_exp.py_mutator import iter_apply_random_mutations
import os
from codetrace.type_inf_exp import py_mutator 
from codetrace.parsing_utils import get_model_fim, placeholder_to_std_fmt
from codetrace.utils import load, save, num_available_devices, get_vllm_config

def filter_incorrect(ds: datasets.Dataset, llm: LLM, new_ds_name, batch_size = 60000) -> datasets.Dataset:
    """
    Filter out examples where the model's prediction is incorrect. Truncate generation and
    solution at 1 token
    """
    params = SamplingParams(temperature=0, max_tokens=1)
    new_ds = []
    model_fim = get_model_fim(get_vllm_config(llm).name_or_path)
    ds = ds.map(lambda x: {
        "prompt" : placeholder_to_std_fmt(x["mutated_program"], model_fim),
        "solution": x["fim_type"]
    }, desc="Prepping prompts")
    
    prompts = ds["prompt"]
    # batch generations so we can save them early
    for b,i in tqdm(
        enumerate(range(0, len(ds), batch_size)), 
        desc="Filtering out incorrect mutations", 
        total=len(ds) // batch_size
    ):
        use_tqdm = (b == 0)
        generations = llm.generate(prompts[i:i+batch_size], params, use_tqdm=use_tqdm)

        for j,output in enumerate(generations):
            generated_text = output.outputs[0].text.strip()
            if generated_text != ds[i+j]["solution"]:
                new_ds.append({**ds[i+j],"mutated_generated_text": generated_text})
                
        # save every
        print(f"Len new_ds: {len(new_ds)}")
        if len(new_ds) > 0:
            new_ds_hf = datasets.Dataset.from_pandas(pd.DataFrame(new_ds))
            save(new_ds_hf, new_ds_name, private=True)
    
    new_ds = datasets.Dataset.from_pandas(pd.DataFrame(new_ds))
    save(new_ds_hf, new_ds_name, private=True)
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

def preprocess_then_mutate(batch, mutations, correct_bool = True):
    post = py_preprocess(batch, correct_bool=correct_bool)
    return iter_apply_random_mutations(post, mutations)

def main(args):
    if "do_mutate" in args.actions:
        ds = load(args.completions_ds, split=args.split)
        if args.max_size > -1:
            ds = ds.shuffle(seed=args.seed).select(range(args.max_size))
        mutations = [getattr(py_mutator, m) for m in args.mutations]
        
        batches = make_batches(ds, cpu_count())
        results = batched_apply(batches, cpu_count(), preprocess_then_mutate, mutations=mutations, correct_bool=args.correct_bool)

        def _yielder():
            for ex in tqdm(results, desc="Yielding", total=len(results)):
                yield ex
                
        ds = datasets.Dataset.from_generator(_yielder)
        print(ds)
        save(ds, args.new_ds_name + "_" + args.model_name + "_unfiltered")
    else:
        ds = load(args.new_ds_name + "_" + args.model_name + "_unfiltered", split=args.split)
        
    if "do_completions" in args.actions:
        tps = num_available_devices()
        print(f"Serving VLLM across {tps} GPUs.")
        llm = LLM(args.model, tensor_parallel_size=tps, tokenizer=args.tokenizer, dtype="bfloat16")
        
        if tps > 1:
            # still want to save some intermediate completions
            batchsize = 1000*tps
        else:
            batchsize = 1000
        ds = filter_incorrect(ds, llm, args.new_ds_name + "_" + args.model_name, batch_size=batchsize)
        print(ds)
        save(ds, args.new_ds_name + "_" + args.model_name)

# TODO remove comments or change pyright settings to ignore commenst (surprisingly, it doesnt)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--completions-ds", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--new-ds-name", type=str, required=True)
    parser.add_argument("--mutations", type=str, required=True, nargs="+", choices=["mutation_rename_type",
                                                                                    "mutation_rename_vars",
                                                                                    "mutation_delete_annotation"])
    parser.add_argument("--gpu", type=int, nargs="+", default=None)
    parser.add_argument("--no-caching", action="store_true", default=False)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--max-size", type=int, default=-1)
    parser.add_argument("--correct-bool", type=bool, default=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--actions", nargs="+", choices=["do_completions", "do_mutate"], default=["do_completions", "do_mutate"])
    args = parser.parse_args()
    if args.no_caching:
        datasets.disable_caching()
        print("Caching enabled?:", datasets.is_caching_enabled())
    print("Gpu:", os.environ["CUDA_VISIBLE_DEVICES"])
    main(args)
