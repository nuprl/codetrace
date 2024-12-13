import subprocess
import argparse
import os
from typing import List, Optional
import shutil
from pathlib import Path
import datasets
from tqdm import tqdm

ALL_MUTATIONS = [
    "vars",
    "types", 
    "delete", 
    "delete_vars_types", 
    "vars_delete", 
    "types_delete", 
    "types_vars"
]

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

def all_subsets(lang:List[str], model:str, mutations:List[str], interval:int, prefix:str):
    subsets = []
    n_layers = model_n_layer(model)
    ranges = list(get_ranges(n_layers, interval))
    for l in lang:
        for r in ranges:
            for m in mutations:
                subsets.append(f"{prefix}{l}-{m}-{r}-{model}")
    return subsets

def _load_parquet(rootdir: str, subset: str, split:str) -> datasets.Dataset:
    path1 =  (Path(rootdir) / subset) / f"{split}-0-of-1.parquet"
    path2 =  (Path(rootdir) / subset) / f"{split}-00000-of-00001.parquet"
    for path in [path1, path2]:
        if path.exists():
            return datasets.Dataset.from_parquet(path.as_posix())
    raise ValueError(f"{rootdir}/{subset}/{split} not found!")

def _postprocess(rootdir:str, subsets:List[str]):
    """
    Need to extract parquet files in downloaded results,
    as well as rename tensors
    """
    # extract parquets
    datasets.disable_caching()
    datasets.disable_progress_bars()
    for subset in tqdm(subsets, desc="extracting data"):
        if (Path(rootdir) / subset).exists():
            for split in ["rand","steer","test"]:
                split_name = split if split != "rand" else "test"
                suffix = "_rand" if split == "rand" else ""
                try:
                    ds = _load_parquet(rootdir, subset, split)
                except FileNotFoundError:
                    pass
                ds.save_to_disk(f"{rootdir}/{subset}/{split_name}_steering_results{suffix}")
    
    # rename tensors
    for tensor in Path(rootdir).glob("*.pt"):
        subset_name = tensor.name.replace(".pt","")
        if subset_name in subsets:
            Path(tensor).rename((Path(rootdir) / subset_name) / "steering_tensor.pt")


def main(
    model: str,
    lang: List[str],
    mutations: List[str],
    prefix: str,
    interval: int,
    token: Optional[str],
    dry_run: bool,
    num_proc: int = 40
):
    auth_token = os.environ.get("HF_AUTH_TOKEN", token)
    FILES = all_subsets(lang, model, mutations, interval, prefix)
    # download results ds
    cmd = [
        "huggingface-cli", "download",
        "--repo-type", "dataset",
        "nuprl-staging/type-steering-results",
        "--include", *[f + "/*.parquet" for f in FILES],
        "--local-dir", "results",
        "--force-download", "--quiet",
        "--max-workers", str(num_proc),
        "--token", auth_token
    ]
    
    if dry_run:
        print("\n".join(cmd))
    else:
        subprocess.run(cmd, check=True)

    # download vectors
    cmd = [
        "huggingface-cli", "download",
        "--repo-type", "model",
        "nuprl-staging/steering-tensors",
        *[f + ".pt" for f in FILES],
        "--local-dir", "results",
        "--token", auth_token
    ]
    
    if dry_run:
        print("\n".join(cmd))
    else:
        subprocess.run(cmd, check=True)

    if not dry_run:
        # post_process
        _postprocess("results", FILES)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",type=str, required=True)
    parser.add_argument("--lang",type=str,nargs="+",choices=["py","ts"],required=True)
    parser.add_argument("--mutations", type=str, nargs="+", choices=ALL_MUTATIONS, default=ALL_MUTATIONS)
    parser.add_argument("--prefix", type=str, required=True, 
                        choices=["steering-","lang_transfer_py_steering-", "lang_transfer_ts_steering-"])
    parser.add_argument("--interval", type=int, choices=[1,3,5], default=True)
    parser.add_argument("--token", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(**vars(args))
    
    