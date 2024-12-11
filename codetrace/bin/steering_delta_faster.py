#!/work/nvme/bcbj/aguha/venv-x86/bin/python3
#SBATCH --job-name=typesteering_steering
#SBATCH --partition=gpuA40x4
#SBATCH --account=bcbj-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=60G
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00

# You must run this script in the codetrace directory and in an environment
# in which it works.

"""
This is the "faster" version because it computes steering tensor for a 
model/mutations steer script ONCE instead of for every layer interval (collects all layers
at once).
"""

import argparse
import subprocess
from pathlib import Path
import datasets
from typing import Optional
import torch

def get_ranges(num_layers: int, interval: int):
    for i in range(0, num_layers):
        if i + interval <= num_layers:
            yield ",".join(map(str, range(i, i + interval)))

"""
Fast utils.

Since for every layer, steering tensor, test split, and steer split are the same
(in the case where all layers are collected during creation of steering tensor),
we can save some time by just loading the previous layer data and saving it
in the current layers' directory.
"""
def save_to_steering_dir(
    steer_split: Optional[datasets.Dataset],
    test_split: Optional[datasets.Dataset],
    steering_tensor: Optional[torch.Tensor],
    output_dir: str
):
    if steer_split != None and not Path(f"{output_dir}/steering_split").exists():
        steer_split.save_to_disk(f"{output_dir}/steering_split")
    if test_split != None and not Path(f"{output_dir}/test_split").exists():
        test_split.save_to_disk(f"{output_dir}/test_split")
    if steering_tensor != None:
        torch.save(steering_tensor, f"{output_dir}/steering_tensor.pt")

def try_load(output_dir:str):
    steer_split, test_split, steering_tensor = None,None,None
    if Path(f"{output_dir}/steering_split").exists():
        steer_split = datasets.load_from_disk(f"{output_dir}/steering_split")
    if Path(f"{output_dir}/test_split").exists():
        test_split = datasets.load_from_disk(f"{output_dir}/test_split")
    if Path(f"{output_dir}/steering_tensor.pt").exists():
        steering_tensor = torch.load(f"{output_dir}/steering_tensor.pt")
    for layer in range(steering_tensor.shape[0]):
        assert steering_tensor[layer].sum().item() != 0, \
            f"Steering tensor layer {layer} should not be empty!"
    return steer_split, test_split, steering_tensor

"""
Main
"""
def main_with_args(model: str, mutations: str, lang: str, num_layers: int, interval: int, dry_run: bool):
    RUN_SPLITS = ["test", "rand","steer"]
    steer_split, test_split, steering_tensor = None,None,None

    for layers in get_ranges(num_layers, interval):
        mutation_underscored = mutations.replace(",", "_")
        layers_underscored = layers.replace(",", "_")
        output_dir = f"results/steering-{lang}-{mutation_underscored}-{layers_underscored}-{model}"
        
        if (Path(output_dir) / "test_results.json").exists() and \
            (Path(output_dir) / "steer_results.json").exists() and \
            (Path(output_dir) / "test_results_rand.json").exists():
            print(f"Skipping {output_dir} because it already exists")
            # load previous steering data. Useful for resuming from existing sweep
            steer_split, test_split, steering_tensor = try_load(output_dir)
            continue

        # save previous steering data in current dir if it exists
        save_to_steering_dir(steer_split, test_split, steering_tensor, output_dir)

        cmd = [
            "python3", "-m", "codetrace.scripts.launch_steer",
            "--model", f"/work/nvme/bcbj/franlucc/models/{model}",
            "--candidates", "nuprl-staging/type-steering",
            "--split", "train", 
            "--subset", f"mutations-{lang}-{mutation_underscored}-{model}",
            "--output-dir", output_dir,
            "--layers", layers,
            "--steer-name", "steering_split",
            "--test-name", "test_split",
            "--tensor-name", "steering_tensor.pt",
            "-n", "3000",
            "--test-size", "100",
            "--collect-all-layers",
            "--run-steering-splits", *RUN_SPLITS,
            "--dedup-type-threshold", "4",
            "--dedup-prog-threshold", "25"
        ]
        
        print(" ".join(cmd))
        if not dry_run:
            subprocess.run(cmd, check=True)

        # load steering data for next layer experiment
        steer_split, test_split, steering_tensor = try_load(output_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--mutations", type=str, required=True)
    parser.add_argument("--lang", type=str, required=True)
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument("--interval", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main_with_args(**vars(args))


if __name__ == "__main__":
    main()
