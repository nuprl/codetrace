import json
import argparse
from pathlib import Path
import datasets
from tqdm import tqdm
from multiprocessing import cpu_count
from codetrace.fast_utils import batched_apply, make_batches
import itertools as it
from collections import defaultdict

ALL_MODELS= ["CodeLlama-7b-Instruct-hf",
            "qwen2p5_coder_7b_base", 
            "Llama-3.2-3B-Instruct",
            "starcoderbase-7b",
            "starcoderbase-1b"]

ALL_MUTS = ["types","vars","delete",
            "vars_delete","types_delete","types_vars",
            "delete_vars_types"]

def model_n_layer(model: str) -> int:
    if "qwen" in model.lower():
        return 28
    elif "codellama" in model.lower():
        return 32
    elif "starcoderbase-1b" in model.lower():
        return 24
    elif "starcoderbase-7b" in model.lower():
        return 42
    elif "llama" in model.lower():
        return 28
    else:
        raise NotImplementedError(f"Model {model} model_n_layer not implemented!")

def get_ranges(num_layers: int, interval: int):
    for i in range(0, num_layers):
        if i + interval <= num_layers:
            yield "_".join(map(str, range(i, i + interval)))

def load(path: Path, **kwargs) -> datasets.Dataset:
    datasets.disable_progress_bars()
    datasets.disable_caching()
    if path.exists():
        parquet = (path / kwargs["name"] / f"{kwargs['split']}-0-of-1.parquet")
        ds= datasets.Dataset.from_parquet(parquet.as_posix())
    else:
        ds = datasets.load_dataset(path.as_posix(), **kwargs)
    return ds

def check_all_generations(path : Path = None):
    if not path:
        path = Path("nuprl-staging/type-steering")
    progress_bar = tqdm(total=(2*len(ALL_MODELS)*len(ALL_MUTS)), desc="Checking generations")
    for lang in ["py","ts"]:
        for model in ALL_MODELS:
            for mut in ALL_MUTS:
                name =f"mutations-{lang}-{mut}-{model}"
                try:
                    ds = load(path=path,name=name,split="train")
                    df = ds.to_pandas()
                    counts_mut = df.value_counts("mutated_generated_text")
                    counts = df.value_counts("generated_text")
                    if counts_mut.get("", 0) > 0:
                        print(f"mutations: {lang}-{mut}-{model}: {counts_mut['']}")
                    if counts.get("", 0) > 0:
                        print(f"generated: {lang}-{mut}-{model}: {counts['']}")
                except FileNotFoundError:
                    print(f"Error file not found {name}")
                progress_bar.update(1)
    progress_bar.close()

def check_all_generations_in_test(
    keydict: list[dict], 
    split="test", 
    path:Path=Path("nuprl-staging/type-steering-results"),
    inner_disable_tqdm=False,
) -> list[dict]:
    assert not keydict or len(keydict) == 1
    if len(keydict) == 1:
        keydict = keydict[0]
    langs=keydict.pop("langs",["py","ts"])
    all_models=keydict.pop("models",ALL_MODELS)
    all_muts=keydict.pop("mutations",ALL_MUTS)

    failed = defaultdict(list)
    progress_bar = tqdm(total=(len(langs)*len(all_models)*len(all_muts)), desc=f"Checking generations in {split}", 
                        disable=inner_disable_tqdm)
    for lang in langs:
        for model in all_models:
            for mut in all_muts:
                for interval in [1,3,5]:
                    for layers in get_ranges(model_n_layer(model), interval):
                        name=f"steering-{lang}-{mut}-{layers}-{model}"
                        try:
                            ds = load(path=path, name=name,split=split,trust_remote_code=True)
                            df = ds.to_pandas()
                            counts_mut = df.value_counts("mutated_generated_text")
                            if counts_mut.get("", 0) > 0:
                                failed[counts_mut['']].append(f"{lang}-{mut}-{layers}-{model}/{split}")
                        except FileNotFoundError:
                            print(f"{split} Error file not found {name}")
                progress_bar.update(1)
    progress_bar.close()
    return [failed]

def multiproc_check_results(split, path: Path=None):
    keys = [{"langs": [l], "models": [llm], "mutations": [m]} 
            for llm,l,m in it.product(ALL_MODELS, ["py","ts"], ALL_MUTS)]
    batches = make_batches(keys, len(keys))
    kwargs = {"split": split, "inner_disable_tqdm":True}
    if path:
        kwargs["path"]=path
    results = batched_apply(batches, len(keys), check_all_generations_in_test, **kwargs)
    print(json.dumps(results, indent=4))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r","--results_dir", type=Path, default=None)
    args = parser.parse_args()
    multiproc_check_results("test", args.results_dir)
    multiproc_check_results("steer", args.results_dir)
    check_all_generations()
    
