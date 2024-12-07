import sys
import datasets

def check_all_generations(ds):
    df = ds.to_pandas()
    counts_mut = df.value_counts("mutated_generated_text")
    counts = df.value_counts("generated_text")
    assert counts_mut.get("", None) is None, f"Found empty mutated generations: {counts_mut['']}"
    assert counts.get("", None) is None, f"Found empty original generations: {counts['']}"

def compare(typechecked, original):
    cols = original.columns
    typechecked = typechecked[cols]
    return len(typechecked.merge(original)) == len(typechecked)

def _cast(x, feats):
    new_x = {}
    for k,v in x.items():
        if k in feats:
            new_x[k] = str(v)
        else:
            new_x[k] = v
    return new_x

def cast_feats(ds):
    list_features = [k for k,v in ds.features.items() if "Sequence" in str(v)]
    return ds.map(lambda x: _cast(x, list_features), num_proc=10, desc="Casting")

if __name__ == "__main__":
    path = sys.argv[1]

    ds = datasets.load_dataset("nuprl-staging/type-steering", name=path.split("/")[-1], split="train")
    ds_typechecked = datasets.load_from_disk(path)
    ds = cast_feats(ds)
    ds_typechecked = cast_feats(ds_typechecked)
    
    print(f"Lost {len(ds) - len(ds_typechecked)} dupes: {len(ds)} -> {len(ds_typechecked)}")

    df = ds.to_pandas()
    df_typechecked = ds_typechecked.to_pandas()
    print(f"Num typechecks {df_typechecked['typechecks'].sum()}/{len(df_typechecked)} = {df_typechecked['typechecks'].mean():.2f}")

    equals = compare(df_typechecked, df)
    assert equals
    print(f"Equality: {equals}")
    assert len(df_typechecked) >= 3000
    print(f"Length >= 3000: {len(df_typechecked) >= 3000}")

    # check if any empty generations
    check_all_generations(df_typechecked)


