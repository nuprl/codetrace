import datasets
from codetrace.type_inf_exp.build_dataset import *
from codetrace.utils import *
from argparse import ArgumentParser
import multiprocessing
from transformers import AutoTokenizer
from tqdm import tqdm
from functools import partial
from typing import Callable

# def _batch_get_prompts(batch, query_str):
#     new_batch = []
#     for ex in batch:
#         prompts = make_natural_typeinf_prompt(ex, query_str)
#         new_batch += prompts
#     return new_batch

# def _batch_remove_comments(batch):
#     new_batch = []
#     for ex in batch:
#         new_batch.append({"content": remove_comments(ex["content"]), **ex})
#     return new_batch

# def _batch_filter_fill_in_prog(batch):
#     new_batch = []
#     for ex in batch:
#         if ": <FILL>" in ex["fim_program"]:
#             new_batch.append(ex)
#     return new_batch


def batch_filter_is_one_token(batch, tokenizer):
    if len(batch) == 0:
        return []
    input_ids = tokenizer.batch_encode_plus([ex["fim_type"] for ex in batch])["input_ids"]
    return [ex for i,ex in enumerate(batch) if len(input_ids[i]) == 1]


def _func_combo(batch, query_str, do_remove_comments):
    """
    func combo for:
    - remove comments (optional)
    - get prompts
    - filter fill in prog
    """
    new_batch = []
    for ex in batch:
        if do_remove_comments:
            new_ex = {"_content": remove_comments(ex["content"]), **ex}
        else:
            new_ex = {"_content": ex["content"], **ex}
            
        prompts = make_natural_typeinf_prompt(new_ex, query_str, content_key="_content")
        
        for p in prompts:
            if ": <FILL>" in p["fim_program"]:
                new_batch.append(p)
    return new_batch

def batched_do_func_from_generator(batches_generator, num_proc, total_len, func, **func_kwargs):
    pool = multiprocessing.Pool(num_proc)
    async_out_batches = []
    for i, batch in tqdm(enumerate(batches_generator), desc="Processing batches", total=total_len):
        async_out = pool.apply_async(func, args=(batch,), kwds=func_kwargs)
        async_out_batches.append(async_out)
    
def batched_do_func(batches, num_proc, func, **func_kwargs):
    pool = multiprocessing.Pool(num_proc)
    
    async_out_batches = []
    for i, batch in tqdm(enumerate(batches), desc="Processing batches", total=len(batches)):
        async_out = pool.apply_async(func, args=(batch,), kwds=func_kwargs)
        async_out_batches.append(async_out)
    
    results = []
    for i in tqdm(range(len(async_out_batches)), desc="Getting results", total=len(async_out_batches)):
        results += async_out_batches[i].get()
            
    pool.close()
    pool.join()
    return results

def get_batches(iterable, len_iter, num_proc):
    batch_size = len_iter // num_proc
    batches = []
    current_batch = []
    for i,ex in tqdm(enumerate(iterable), desc="Making batches", total=len_iter):
        if len(current_batch) == batch_size:
            batches.append(current_batch)
            current_batch = []
        current_batch.append(ex)
    return batches

 
def _collect_index(itr, si, ei):
    return itr[si:ei]
       
def get_batches_fast(iterable, len_iter, num_proc):
    """
    Not for IterableDataset
    """
    batch_size = len_iter // num_proc
    pool = multiprocessing.Pool(num_proc)
    async_out_batches = []
    
    i=0
    progress_bar = tqdm(total=num_proc, desc="Making batches")
    while i < len_iter:
        end_index = min(i + batch_size, len_iter)
        async_out = pool.apply_async(_collect_index, args=(iterable, i, end_index))
        async_out_batches.append(async_out)
        i = end_index
        progress_bar.update(1)
    progress_bar.close()
    
    batches = []
    for j in tqdm(range(len(async_out_batches)), desc="Getting results", total=len(async_out_batches)):
        batches.append(async_out_batches[j].get())
    return batches

def yield_batches_fast(iterable, len_iter, num_proc):
    """
    Not for IterableDataset
    """
    batch_size = len_iter // num_proc
    pool = multiprocessing.Pool(num_proc)
    async_out_batches = []
    
    i=0
    while i < len_iter:
        end_index = min(i + batch_size, len_iter)
        async_out = pool.apply_async(_collect_index, args=(iterable, i, end_index))
        async_out_batches.append(async_out)
        i = end_index
    
    for j in range(len(async_out_batches)):
        yield async_out_batches[j].get()

def multi_process(ds, args):
    """
    Method for processing a dataset with multiple processes.
    Does following:
    - removes comments (optional)
    - gets natural typeinf prompts
    - filters out examples that do not have a fill in program
    - pushes to hub
    - pushes a copy to hub where all examples have a one token type
    
    All operations are batched and done in parallel. Faster than huggingface multiproc map/filter.
    """
    batches = get_batches_fast(ds, args.len_ds, args.num_proc)
    del ds
    results = batched_do_func(batches, args.num_proc, _func_combo, query_str=TS_QUERY_FUNC_TYPES, do_remove_comments=args.do_remove_comments)
    ds = datasets.Dataset.from_pandas(pd.DataFrame(results))
    print(ds)
    ds.push_to_hub(args.output_ds)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    batches = get_batches_fast(results, len(results), args.num_proc)
    ds = batched_do_func(batches, args.num_proc, _batch_filter_is_one_token, tokenizer=tokenizer)
    ds = datasets.Dataset.from_pandas(pd.DataFrame(ds))
    ds.push_to_hub(args.output_ds + "-1tok")
    print(ds)

def process(ds, args):
    """
    Method for processing a dataset with a single process.
    Does following:
    - removes comments (optional)
    - gets natural typeinf prompts
    - filters out examples that do not have a fill in program
    - pushes to hub
    - pushes a copy to hub where all examples have a one token type
    """
    if args.remove_comments:
        ds = ds.map(lambda x: {"content": remove_comments(x["content"])})

    def _generate_prompts(ds):
        for ex in ds:
            prompts = make_natural_typeinf_prompt(ex, TS_QUERY_FUNC_TYPES)
            for prompt in prompts:
                yield prompt
    
    prompts = datasets.Dataset.from_generator(partial(_generate_prompts, ds), features=ds.features)
        
    ds = ds.filter(lambda x: ": <FILL>" in x["fim_program"])
    print(ds)
    ds.push_to_hub(args.output_ds)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    ds = ds.filter(lambda x: len(tokenizer(x["fim_type"])["input_ids"]) == 1, desc="Creating one token ds")
    ds.push_to_hub(args.output_ds + "-1tok")
    print(ds)

def _get_subset(iterable_ds, max_size):
    for i, ex in enumerate(iterable_ds):
        if i >= max_size:
            break
        yield ex

def main(args):
    if not args.do_multiproc:
        print("Using single process")
        ds = datasets.load_dataset(args.input_ds, split="train")

        if args.max_size > -1:
            ds = ds.select(range(args.max_size))
    
        process(ds, args)
    else:
        print("Using multiprocessing")
        ds = datasets.load_dataset(args.input_ds, split="train")
        len_ds = ds.info.splits["train"].num_examples
        if args.max_size > 0:
            len_ds = min(len_ds, args.max_size)
            ds = datasets.IterableDataset.from_generator(_get_subset, gen_kwargs={"iterable_ds": ds, "max_size": args.max_size})
        
        print("Length of dataset: ", len_ds)
        args.len_ds = len_ds
        args.num_proc = multiprocessing.cpu_count()
        multi_process(ds, args)
        
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--output_ds", type=str, required=True)
    parser.add_argument("--input_ds", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default="/home/arjun/models/starcoderbase-1b")
    parser.add_argument("--do-remove-comments", action="store_true", default=False)
    parser.add_argument("--max-size", type=int, default=-1)
    parser.add_argument("--do-multiproc", action="store_true", default=False)
    args = parser.parse_args()
    main(args)