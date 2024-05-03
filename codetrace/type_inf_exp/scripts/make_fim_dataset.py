import datasets
from argparse import ArgumentParser
import multiprocessing
from transformers import AutoTokenizer
from tqdm import tqdm
from functools import partial
from typing import Callable
from codetrace.fast_utils import get_batches_fast, batched_do_func
from typing import List
import os
from pathlib import Path
from codetrace.utils import lang_to_builder, lang_to_parser, replace_between_bytes, get_captures

TS_QUERY_FUNC_TYPES = """
(required_parameter pattern: (_) (type_annotation) @tp)
(optional_parameter pattern: (_) (type_annotation) @tp)
return_type: (type_annotation) @tp
"""

def make_natural_typeinf_prompt(
    ex : dict,
    type_annotation_query : str,
    content_key : str = "content"
) -> List[dict]:
    """
    Make a dataset where:
    - for each program, make prompts for every type annotation
    """
    prompts = []
    type_annotations = get_captures(ex[content_key], type_annotation_query, language="ts")
    for c in type_annotations:
        byte_fim_prog = replace_between_bytes(bytes(ex[content_key], "utf-8"), c[0].start_byte, c[0].end_byte, ": <FILL>")
        fim_program = byte_fim_prog.decode("utf-8").strip()
        fim_type = c[0].text.decode("utf-8").replace(":","").strip()
        prompts.append({"fim_program": fim_program, "fim_type": fim_type, **ex})

    return prompts

def batch_filter_is_one_token(batch, tokenizer):
    if len(batch) == 0:
        return []
    input_ids = tokenizer.batch_encode_plus([ex["fim_type"] for ex in batch])["input_ids"]
    return [ex for i,ex in enumerate(batch) if len(input_ids[i]) == 1]

def remove_comments(program : str) -> str:
    comment_query = """((comment) @comment)"""
    language = "ts"
    lang = lang_to_builder[language]
    parser = lang_to_parser[language]
    comment_query = lang.query(comment_query)
    tree = parser.parse(bytes(program, "utf8"))
    captures = comment_query.captures(tree.root_node)
    # sort by start byte descending
    captures.sort(key=lambda x: x[0].start_byte, reverse=True)
    program = tree.text
    for c in captures:
        program = replace_between_bytes(program, c[0].start_byte, c[0].end_byte, "")
    return program.decode("utf-8").strip()

def _process_prompts(batch, query_str, do_remove_comments):
    """
    func combo for:
    - remove comments (optional)
    - get prompts
    - filter fill in prog
    """
    new_batch = []
    for ex in batch:
        # filter out too large or too small
        if ex["size"] > 10000 or ex["size"] < 1000:
            continue
        
        if do_remove_comments:
            new_ex = {"_content": remove_comments(ex["content"]), **ex}
        else:
            new_ex = {"_content": ex["content"], **ex}
        
        prompts = make_natural_typeinf_prompt(new_ex, query_str, content_key="_content")
        
        for p in prompts:
            if ": <FILL>" in p["fim_program"]:
                new_batch.append(p)
    return new_batch

    
def multiprocess(ds, args):
    """
    Method for processing a dataset with multiple processes.
    Does following:
    - removes comments (optional)
    - gets natural typeinf prompts
    - filters out examples that do not have a fill in program
    - pushes to hub
    - pushes a copy to hub where all examples have a one token type
    
    NOTE: chunking is necessary because filtering 1-tok is expensive once
    all possible fim_types are generated.
    """
    for i in range(args.num_chunks):
        print(f"Processing chunk {i} / {args.num_chunks}")
        ds_chunk=ds.shard(args.num_chunks, i)
        batches = get_batches_fast(ds_chunk, args.num_proc)
        del ds_chunk
        results = batched_do_func(batches, args.num_proc, _process_prompts, query_str=TS_QUERY_FUNC_TYPES, do_remove_comments=args.do_remove_comments)
        print(f"Length of ds: {len(results)}")
        def yielder():
            for ex in tqdm(results, desc="Yielding", total=len(results)):
                yield ex
        ds = datasets.Dataset.from_generator(yielder)
        print(ds)
        if args.num_chunks == 1:
            ds.push_to_hub(args.output_ds + "-ntok", private=True)
        else:
            ds.save_to_disk(f"{args.cache_dir}/"+ args.output_ds.split("/")[-1] + f"-chunk_{i}")

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        batches = get_batches_fast(results, args.num_proc)
        ds = batched_do_func(batches, args.num_proc, batch_filter_is_one_token, tokenizer=tokenizer)
        print(f"Length of ds: {len(ds)}")
        def yielder():
            for ex in tqdm(ds, desc="Yielding", total=len(ds)):
                yield ex
        ds = datasets.Dataset.from_generator(yielder)
        print(ds)
        if args.num_chunks == 1:
            ds.push_to_hub(args.output_ds + "-1tok", private=True)
            return
        ds.save_to_disk(f"{args.cache_dir}/"+ args.output_ds.split("/")[-1] + f"-1tok-chunk_{i}")
        
    # load all chunks and push to hub
    ds_1tok = []
    ds_any_tok = []
    for i in range(args.num_chunks):
        ds = datasets.load_from_disk(f"{args.cache_dir}/"+ args.output_ds.split("/")[-1] + f"-chunk_{i}")
        one_tok = datasets.load_from_disk(f"{args.cache_dir}/"+ args.output_ds.split("/")[-1] + f"-1tok-chunk_{i}")
        ds_1tok.append(one_tok)
        ds_any_tok.append(ds)
    ds_1tok = datasets.concatenate_datasets(ds_1tok)
    print(ds_1tok)
    ds_1tok.push_to_hub(args.output_ds + "-1tok", private=True)
    
    ds_any_tok = datasets.concatenate_datasets(ds_any_tok)
    print(ds_any_tok)
    ds_any_tok.push_to_hub(args.output_ds + "-ntok", private=True)

    
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

    def generate_prompts():
        for ex in ds:
            prompts = make_natural_typeinf_prompt(ex, TS_QUERY_FUNC_TYPES)
            for prompt in prompts:
                yield prompt
    
    prompts = datasets.Dataset.from_generator(generate_prompts, features=ds.features)
        
    ds = ds.filter(lambda x: ": <FILL>" in x["fim_program"])
    print(ds)
    ds.push_to_hub(args.output_ds)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    ds = ds.filter(lambda x: len(tokenizer(x["fim_type"])["input_ids"]) == 1, desc="Creating one token ds")
    ds.push_to_hub(args.output_ds + "-1tok")
    print(ds)


def main(args):
    ds = datasets.load_dataset(args.input_ds, split="train")

    if args.max_size > -1:
        ds = ds.shuffle().select(range(args.max_size))
            
    if not args.do_multiproc:
        print("Using single process")
        process(ds, args)
    else:
        print("Using multiprocessing")
        if os.path.exists(Path(args.cache_dir)):
            raise ValueError(f"Cache dir {args.cache_dir} already exists. Please remove.")
        os.mkdir(args.cache_dir)
        args.num_proc = multiprocessing.cpu_count()
        multiprocess(ds, args)
        
        
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--output-ds", type=str, required=True)
    parser.add_argument("--input-ds", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--do-remove-comments", action="store_true", default=False)
    parser.add_argument("--max-size", type=int, default=-1)
    parser.add_argument("--do-multiproc", action="store_true", default=False)
    parser.add_argument("--cache-dir", type=str, default="dataset_chunks")
    parser.add_argument("--num-chunks", type=int, default=1)
    args = parser.parse_args()
    main(args)