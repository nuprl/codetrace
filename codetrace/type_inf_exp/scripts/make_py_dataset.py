import datasets
from codetrace.type_inf_exp.build_dataset import py_remove_annotations
from multiprocessing import cpu_count

ds = datasets.load_dataset("franlucc/fim_items_10k", split="train")
fim_placeholder="_$FILL"
# make a fim_program and fim_type column
ds = ds.map(lambda x: {"fim_program": x["prefix"] + fim_placeholder + x["suffix"], 
                       "fim_type": x["middle"].strip(), **x}, num_proc=cpu_count())

# map py_remove_annotations to fim_program
ds = ds.map(lambda x: {"fim_program": py_remove_annotations(x["fim_program"], fim_placeholder)}, num_proc=cpu_count())

# replace with standard placeholder
ds = ds.map(lambda x: {"fim_program": x["fim_program"].replace(fim_placeholder, "<FILL>")}, num_proc=cpu_count())
# save
ds.push_to_hub("franlucc/fim_items_10k_prompt_vanilla")