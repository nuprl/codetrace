import datasets
import argparse
from codetrace.utils import placeholder_to_std_fmt, STARCODER_FIM, get_captures
from codetrace.fast_utils import get_batches_fast, batched_do_func
from multiprocessing import cpu_count
from tqdm import tqdm
from collections import Counter

def get_bad_type(program, gold, language):
    if language == "py":
        query = """(typed_parameter) @param"""
    else:
        query = """(type_identifier) @param"""
        
    types = get_captures(program, query, language=language)
    for c in types:
        c = c[0]
        text = c.text.decode("utf-8").replace(":","").strip()
        if text != gold:
            return text
    
    if language == "py":
        predefined = ["str","int"]
    else:
        predefined = ["string","number"]
    for typ in predefined:
        if typ != gold:
            return typ
    return None

def transform(batch, language):
    new_items = []
    for item in batch:
        prompt = placeholder_to_std_fmt(item["fim_program"], STARCODER_FIM)
        gold = item["fim_type"]
        fim_program = prompt+gold
        neg = get_bad_type(prompt.replace("<FILL>",gold), gold, language)
        if neg == None:
            continue
        mutated = prompt + neg
        new_items.append({**item, "fim_program":fim_program, "mutated_program": mutated})
    return new_items

def main(args):
    ds = datasets.load_dataset(args.src_ds, split="train")
    batches = get_batches_fast(ds, len(ds), cpu_count())
    results = batched_do_func(batches, cpu_count(), transform, language=args.lang)
    
    def yielder():
        for ex in tqdm(results, desc="Yielding", total=len(results)):
            yield ex
            
    new_ds = datasets.Dataset.from_generator(yielder)
    new_ds.push_to_hub(args.dst_ds, private=True)
    counts = Counter([i["mutated_program"].split("<fim_middle>")[-1].strip() for i in new_ds])
    print(counts)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_ds", type=str, required=True)
    parser.add_argument("--dst_ds", type=str, required=True)
    parser.add_argument("--lang", type=str, required=True, choices=["py","ts"])
    args = parser.parse_args()
    main(args)