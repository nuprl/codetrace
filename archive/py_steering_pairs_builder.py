from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import datasets
import codetrace.py_mutator as py_mutator
import json
from dataclasses import dataclass
from typing import Generator
import argparse
from pathlib import Path
from tqdm.auto import tqdm

@dataclass
class SteeringCandidate:
    target: str
    generator: Generator[str, None, None]
    last_positive_prompt: str

    def __init__(self, item, apply_all_mutations):
        self.target = item["middle"]
        pos_context = item["prefix"] + item["middle"] + item["suffix"]
        prefix_length = len(item["prefix"])
        self.generator = py_mutator.random_mutations(pos_context, prefix_length, apply_all_mutations)
        self.last_positive_prompt = f"<fim_prefix>{item['prefix']}<fim_suffix>{item['suffix']}<fim_middle>"

    def candidate_neg_prompt(self):
        try:
            gen = next(self.generator)
            (new_target_index, candidate_code) = gen
            # print(new_target_index, candidate_code[new_target_index:new_target_index + len(self.target)])
            suffix_start = new_target_index + len(self.target)
            prefix = candidate_code[:new_target_index]
            suffix = candidate_code[suffix_start:]
            return f"<fim_prefix>{prefix}<fim_suffix>{suffix}<fim_middle>"
        except StopIteration:
            return None

def get_next_tokens(model, tokenizer, prompts):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    model.eval()
    with torch.no_grad():
        outputs = model.forward(**inputs)
        # All items in batch, last token in sequence, all logits
    last_token_id_logits = outputs.logits[:, -1, :]
    last_token_id_dists = torch.softmax(last_token_id_logits, dim=1)
    last_token_ids = torch.argmax(last_token_id_dists, dim=1)
    last_token_ids = last_token_ids.to("cpu").tolist()
    last_tokens = [ tokenizer.decode(token) for token in last_token_ids ]
    return last_tokens


def generate_fim_items(code_dataset):
    for item in tqdm(code_dataset, desc="Generating FIM items"):
        contents = item["contents"]
        contents_bytes = contents.encode("utf-8")
        type_annotations = item["type_annotations"]
        type_annotation_starts = item["type_annotation_starts"]
        type_annotation_ends = item["type_annotation_ends"]
        for type_index in range(len(type_annotations)):
            prefix_bytes = contents_bytes[:type_annotation_starts[type_index]]
            suffix_bytes = contents_bytes[type_annotation_ends[type_index]:]
            middle_bytes = contents_bytes[type_annotation_starts[type_index]:type_annotation_ends[type_index]]
            key = f"{item['zip']}/{item['filename']}"
            yield {
                "key": key,
                "prefix": prefix_bytes.decode("utf-8"),
                "suffix": suffix_bytes.decode("utf-8"),
                "middle": middle_bytes.decode("utf-8")
            }


def filter_immediate_mispredictions(model, tokenizer, batch_size, code_dataset):
    """
    The model may immediately mispredict the next code for some code items.
    This is a filter to remove those items.
    """

    results = [ ]
    for ix in range(0, len(code_dataset), batch_size):
        batch = code_dataset[ix:ix + batch_size]
        prompts = [ f"<fim_prefix>{item['prefix']}<fim_suffix>{item['suffix']}<fim_middle>" for item in batch ]
        next_tokens = get_next_tokens(model, tokenizer, prompts)
        for (item, next_token) in zip(batch, next_tokens):
            if item["middle"].strip() == next_token.strip():
                results.append(item)
    return results


def generate_steering_pairs(model, tokenizer, batch_size, code_dataset, apply_all_mutations):
    # In case we are working with a tiny dataset.
    if len(code_dataset) < batch_size:
        batch_size = len(code_dataset)
    
    code_items_iter = iter(generate_fim_items(code_dataset))

    # Initializing the first batch:

    # (1/3). The dataset: each has the fields "prefix", "middle", "suffix"
    code_items =  [ next(code_items_iter) for _ in range(batch_size) ]
    code_items = filter_immediate_mispredictions(model, tokenizer, batch_size, code_items)
    results = [ ]

    # (2/3) The positive prompts, where the model predicts item.middle as the
    # most likely next token.
    candidates = [ SteeringCandidate(item, apply_all_mutations) for item in code_items ]

    while True:
        # (3/3) A batch of mutations to each positive prompt.
        prompts = [ item.candidate_neg_prompt() for item in candidates ]

        # It is possible that no more mutations are possible, so we filter them out.
        none_indices = [ i for i, prompt in enumerate(prompts) if prompt is None ]
        prompts = [ prompts[i] for i in range(len(prompts)) if i not in none_indices ]
        # We also filter out the positive prompts that have no more mutations.
        candidates = [ candidates[i] for i in range(len(candidates)) if i not in none_indices ]

        # We only try to predict the next token if there are any candidates mutants.
        # Without this check we get a tokenizer error.            
        if len(candidates) > 0:
            last_tokens = get_next_tokens(model, tokenizer, prompts)
            
            # We now partition the negative prompts. If a prompt actually leads
            # to a mis-prediction, then we add it to results. If not, we
            # keep it in the candidates list for the next iteration.                    
            next_candidates = [ ]
            for (actual, candidate, candidate_neg_prompt) in zip(last_tokens, candidates, prompts):
                if not candidate.target.strip().startswith(actual.strip()):
                    results.append({
                        "target": candidate.target,
                        "positive": candidate.last_positive_prompt,
                        "negative": candidate_neg_prompt,
                    })
                else:
                    candidate.last_positive_prompt = candidate_neg_prompt
                    next_candidates.append(candidate)
            candidates = next_candidates
        
        # We fill up the batch with new candidates. But, it requires a little care.
        # (1/3). We pull items from code_items_iter, taking care to stop when it is exhausted.
        new_candidates = [ ] 
        for _ in range(batch_size - len(candidates)):
            try:
                new_candidates.append(next(code_items_iter))
            except StopIteration:
                break        
        # (2/3). We filter out items that are immediately mispredicted.
        new_candidates = filter_immediate_mispredictions(model, tokenizer, batch_size, new_candidates)
        # (3/3). We convert the new candidates to SteeringCandidates.
        for item in new_candidates:
            candidates.append(SteeringCandidate(item, apply_all_mutations))

        # We may yield [ ], but there may still be more to yield later.
        yield results
        results = [ ]

        if len(candidates) == 0:
            return


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--model", type=str, required=True)
    args.add_argument("--output", type=Path, required=True)
    args.add_argument("--batch-size", type=int, default=50)
    args.add_argument("--start-index", type=int)
    args.add_argument("--limit", type=int)
    args.add_argument("--all-mutations", action="store_true")
    args = args.parse_args()

    assert (args.start_index is None and args.limit is None) or (args.start_index is not None and args.limit is not None)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        use_cache=True,
        attn_implementation="flash_attention_2"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    ds = datasets.load_dataset("nuprl/manytypes4py", split="train")
    ds = ds.filter(lambda item: (len(item["contents"]) < 2048 * 3) and len(item["type_annotations"]) > 0)
    if args.start_index is not None:
        ds = ds.select(range(args.start_index, min(len(ds), args.start_index + args.limit)))

    gen = generate_steering_pairs(model, tokenizer, args.batch_size, ds, args.all_mutations)    

    with open(args.output, "w") as f:
        counter = 0
        for batch in gen:
            counter += len(batch)
            tqdm.write(f"Produced {counter} steering pairs")
            for item in batch:
                f.write(json.dumps(item))
                f.write("\n")
            f.flush()

if __name__ == "__main__":
    main()
