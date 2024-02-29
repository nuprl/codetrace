import datasets
from codetrace.type_inf_exp.scripts.rename_vars import *
from codetrace.utils import *
from argparse import ArgumentParser
from multiprocessing import cpu_count
from transformers import AutoTokenizer

"""
Take a humaneval/mbpp MultiPLE style dataset. Chain together prompt and canonical solution.
Cut prompt in half (half of tokens in the function). Inspired by https://arxiv.org/pdf/2402.05980.pdf

TODO codegen prompts fim or no fim?
"""
parser = ArgumentParser()
parser.add_argument("--output_ds", type=str, required=True)
parser.add_argument("--input_ds", type=str, required=True)
parser.add_argument("--prompt_column", type=str, required=True)
parser.add_argument("--code_column", type=str, required=True)
parser.add_argument("--split", type=str, required=True)
parser.add_argument("--tokenizer", type=str, default="/home/arjun/models/starcoderbase-1b")
args = parser.parse_args()

ds = datasets.load_dataset(args.input_ds, split=args.split)
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

def halve_code(code):
    tokens = tokenizer.tokenize(code)
    half = len(tokens) // 2
    tokens = tokenizer.convert_tokens_to_string(tokens[:half])
    return tokens

ds = ds.map(lambda x: {"code_prompt": halve_code(x[args.code_column])})

# make a codegen prompt by concatenation args.prompt_column and code_prompt
ds = ds.map(lambda x: {"codegen_prompt": x[args.prompt_column] + "\n" + x["code_prompt"]})

print(ds)
ds.push_to_hub(args.output_ds)


# make new MultiPLE benchmark

'''
def my_function(a: int, b: int, c: int, k: int) -> int:
    """
    Given positive integers a, b, and c, return an integer n > k such that
    (a ** n) + (b ** n) = (c ** n).
    """
    pass
    

### Unit tests below ###
def check(candidate):
    assert candidate(1, 1, 2, 0) == 1
    assert candidate(3, 4, 5, 0) == 2

def test_check():
    check(my_function)
'''

args.output_ds = args.output_ds.split("/")[-1]
os.makedirs(f"exp_data/{args.output_ds}", exist_ok=True)
typing = "from typing import *\n"
for ex in ds:
    with open(f"exp_data/{args.output_ds}/{ex['name']}.py", "w") as f:
        f.write(ex["codegen_prompt"] + "\n\n### Unit tests below ###\n" + ex["tests"])
