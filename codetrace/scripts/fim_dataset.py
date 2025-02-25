import datasets
from argparse import ArgumentParser
import multiprocessing
from transformers import AutoTokenizer,PreTrainedTokenizer
from tqdm import tqdm
from codetrace.fast_utils import make_batches, batched_apply
from typing import List,Dict,Optional,Union
import os
from pathlib import Path
from codetrace.utils import load_dataset,save_dataset
from codetrace.parsing_utils import TS_LANGUAGE, get_captures, replace_between_bytes

TS_QUERY_FUNC_TYPES = """
(required_parameter pattern: (_) (type_annotation) @tp)
(optional_parameter pattern: (_) (type_annotation) @tp)
return_type: (type_annotation) @tp
"""

def build_fim_dataset(
    data: List[Dict], 
    type_annotation_query:str, 
    do_remove_comments:bool
)->List[Dict]:
    """
    func combo for:
    - remove comments (optional)
    - get prompts
    - filter fill in prog
    """
    new_batch = []
    for ex in data:
        if do_remove_comments:
            new_ex = {"_content": remove_comments(ex["content"]), **ex}
        else:
            new_ex = {"_content": ex["content"], **ex}
        
        prompt_dicts = build_type_inference_prompts(new_ex["_content"], type_annotation_query)
        
        for pdict in prompt_dicts:
            if ": <FILL>" in pdict["fim_program"]:
                new_batch.append({**ex, **pdict})
    return new_batch

# filter by 1 token answer
def filter_1tok(batch:List[str], tokenizer) -> List[str]:
    new_batch = []
    for b in batch:
        if len(tokenizer(b["fim_type"], add_special_tokens=False)["input_ids"]) == 1:
            new_batch.append(b)
    return new_batch

def build_type_inference_prompts(prompt: str, type_annotation_query: str) -> List[Dict[str,str]]:
    """
    Make a dataset where:
    - for each program, make prompts for every type annotation
    """
    fim_prompts = []
    type_annotations = get_captures(prompt, type_annotation_query, language="ts")
    for c in type_annotations:
        byte_fim_prog = replace_between_bytes(bytes(prompt, "utf-8"), c[0].start_byte, c[0].end_byte, ": <FILL>")
        fim_program = byte_fim_prog.decode("utf-8").strip()
        fim_type = c[0].text.decode("utf-8").replace(":","").strip()
        fim_prompts.append({"fim_program": fim_program, "fim_type": fim_type})

    return fim_prompts

def remove_comments(program: str) -> str:
    comment_query = """((comment) @comment)"""
    comment_query = TS_LANGUAGE.query(comment_query)
    tree = parser.parse(bytes(program, "utf8"))
    captures = comment_query.captures(tree.root_node)
    # sort by start byte descending
    captures.sort(key=lambda x: x[0].start_byte, reverse=True)
    program = tree.text
    for c in captures:
        program = replace_between_bytes(program, c[0].start_byte, c[0].end_byte, "")
    return program.decode("utf-8").strip()

def main(
    ds: datasets.Dataset,
    tokenizer: PreTrainedTokenizer,
    output_ds: str,
    num_chunks: int,
    nproc: int,
    do_remove_comments: bool
):  
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
    for i in range(num_chunks):
        print(f"Processing chunk {i} / {num_chunks}")
        batches = make_batches(ds.shard(num_chunks, i), nproc)
        results = batched_apply(
            batches, 
            nproc, 
            build_fim_dataset, 
            query_str=TS_QUERY_FUNC_TYPES, 
            do_remove_comments=do_remove_comments
        )
        print(f"Length of ds: {len(results)}")
        ds = datasets.Dataset.from_list(results)
        save_dataset(ds, f"{output_ds}-chunk_{i}")

        batches = make_batches(results, nproc)
        ds_1tok = batched_apply(batches, nproc, filter_1tok, tokenizer=tokenizer)
        print(f"Length of ds: {len(ds_1tok)}")
        ds_1tok = datasets.Dataset.from_list(ds_1tok)
        save_dataset(ds_1tok, f"{output_ds}-1tok-chunk_{i}")
        
    # load all chunks and push to hub
    chunks, chunks_1tok = [],[]
    for i in range(args.num_chunks):
        data = load_dataset(f"{output_ds}-chunk_{i}")
        data_1tok = load_dataset(f"{output_ds}-1tok-chunk_{i}")
        chunks.append(data)
        chunks_1tok.append(data_1tok)

    ds_1tok = datasets.concatenate_datasets(chunks_1tok)
    save_dataset(output_ds + "-1tok")
    
    ds = datasets.concatenate_datasets(chunks)
    save_dataset(output_ds + "-ntok")
        
        
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--output-ds", type=str, required=True)
    parser.add_argument("--input-ds", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--nproc", type=int, required=True)

    parser.add_argument("--do-remove-comments", action="store_true", default=False)
    parser.add_argument("--max-size", type=int, default=-1)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--split", type=str, default=None)
    args = parser.parse_args()
    if args.nproc == -1:
        args.nproc = multiprocessing.cpu_count()

    ds = datasets.load_dataset(args.input_ds, split=args.split)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if args.max_size > -1:
        ds = ds.shuffle().select(range(args.max_size))

    main(ds, tokenizer, args.output_ds, args.num_chunks, args.nproc, args.do_remove_comments)