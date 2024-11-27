import sys
import datasets
import pandas as pd

ds = datasets.load_dataset("nuprl-staging/type-steering", name=sys.argv[1].split("/")[-1], split="train")
ds_typechecked = datasets.load_from_disk(sys.argv[1])

print(f"Lost {len(ds) - len(ds_typechecked)} dupes: {len(ds)} -> {len(ds_typechecked)}")

df = ds.to_pandas()
df_typechecked = ds_typechecked.to_pandas()
print(f"Num typechecks {df_typechecked['typechecks'].sum()}/{len(df_typechecked)} = {df_typechecked['typechecks'].mean():.2f}")

def compare(typechecked, original):
    cols = original.columns
    typechecked["mutation_names"] = typechecked["mutation_names"].astype(str)
    original["mutation_names"] = original["mutation_names"].astype(str)
    typechecked = typechecked[cols]
    return len(typechecked.merge(original)) == len(typechecked)
    

equals = compare(df_typechecked, df)
assert equals
print(equals)
assert len(df_typechecked) >= 3000

