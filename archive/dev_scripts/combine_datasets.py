from tqdm import tqdm
import datasets
import sys


py_dataset_fmt = "franlucc/py_<EDIT>_may3_seed<SEED>_starcoderbase-1b_typechecked"
print(py_dataset_fmt)

edits = [
    "delete_annotation",
    "rename_types",
    "rename_vars",
    "all_mutations",
    "all_renames",
    "types_and_delete",
    "vars_and_delete"
]
def dedup(ds, key):
    ds = ds.shuffle()
    seen = set()
    for i,ex in enumerate(ds):
        if ex[key] in seen:
            continue
        seen.add(ex[key])
        yield ex
        
for edit in tqdm(edits):
    py_edit = py_dataset_fmt.replace("<EDIT>", edit)
    ds0 = py_edit.replace("<SEED>", "0")
    ds1 = py_edit.replace("<SEED>", "1")
    ds0 = datasets.load_dataset(ds0, split="train")
    ds1 = datasets.load_dataset(ds1, split="train")
    concat = datasets.concatenate_datasets([ds0, ds1]).shuffle()
    print(ds0,ds1,concat)
    def my_dedup():
        return dedup(concat, "mutated_program")
    
    dedup_ds = datasets.Dataset.from_generator(my_dedup)
    print(dedup_ds)
    new_name = py_edit.replace("<SEED>", "-0-1")
    dedup_ds.push_to_hub(new_name)