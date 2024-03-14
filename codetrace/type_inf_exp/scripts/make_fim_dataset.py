import datasets
from codetrace.type_inf_exp.build_dataset import *
from codetrace.utils import *
from argparse import ArgumentParser
from multiprocessing import cpu_count
from transformers import AutoTokenizer

parser = ArgumentParser()
parser.add_argument("--output_ds", type=str, required=True)
parser.add_argument("--input_ds", type=str, required=True)
parser.add_argument("--tokenizer", type=str, default="/home/arjun/models/starcoderbase-1b")

args = parser.parse_args()

ds = datasets.load_dataset(args.input_ds, split="train")
ds = ds.map(lambda x: {"content": remove_comments(x["content"])}, num_proc=cpu_count())

ds = make_typeinf_prompts(ds, QUERY_FUNC_TYPES)
ds.push_to_hub(args.output_ds + "_prefilter")
ds = ds.filter(lambda x: ": <FILL>" in x["fim_program"], num_proc=cpu_count())
print(ds)
ds.push_to_hub(args.output_ds)

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

ds = ds.filter(lambda x: len(tokenizer(x["fim_type"])["input_ids"]) == 1, num_proc=cpu_count())
ds.push_to_hub(args.output_ds + "-1tok")