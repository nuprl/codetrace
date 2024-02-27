import datasets
from codetrace.type_inf_exp.build_dataset import *
from codetrace.utils import *
from argparse import ArgumentParser
from multiprocessing import cpu_count

parser = ArgumentParser()
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--ds", type=str, required=True)
args = parser.parse_args()

ds = datasets.load_dataset(args.ds, split="train")

ds = ds.filter(lambda x: len(x["content"]) > 500 and len(x["content"]) < 10000, num_proc=cpu_count())
ds = ds.map(lambda x: {"content": remove_comments(x["content"])}, num_proc=cpu_count())

ds = make_typeinf_prompts(ds)
ds = ds.filter(lambda x: "<FILL>" in x["fim_program"], num_proc=cpu_count())
print(ds)
ds.push_to_hub(args.output_dir)