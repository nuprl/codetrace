#!/work/nvme/bcbj/aguha/venv-x86/bin/python3
#SBATCH --job-name=custom_typesteering_steering
#SBATCH --partition=gpuA40x4
#SBATCH --account=bcbj-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=60G
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00

from pathlib import Path
import argparse
import sys
import os
import torch
from typing import Optional
import datasets
import subprocess
import shutil

DESCRIPTION = """
Start steering with a precomputed steering tensor and test set (already preprocessed, deduplicated,
typechecked etc.)
""".strip()

def get_ranges(num_layers: int, interval: int):
    for i in range(0, num_layers):
        if i + interval <= num_layers:
            yield ",".join(map(str, range(i, i + interval)))

def _run_layer_steer(
    num_layers:int, 
    interval:int, 
    mutations:str, 
    lang:str, 
    model:str, 
    results_prefix:str, 
    steering_field:str,
    dry_run:bool
):
    # don't run steer, since steering tensor is precomputed
    # don't run rand, since we already have the data
    RUN_SPLITS = ["test"]
    
    for layers in get_ranges(num_layers, interval):
        mutation_underscored = mutations.replace(",", "_")
        layers_underscored = layers.replace(",", "_")
        output_dir = f"results/{results_prefix}steering-{lang}-{mutation_underscored}-{layers_underscored}-{model}"
        
        assert (Path(output_dir) / "steering_tensor.pt").exists(), \
            f"Expected existing tensor {(Path(output_dir) / 'steering_tensor.pt').as_posix()}"
        assert (Path(output_dir) / "test_split").exists(), \
            f"Excpected existing test {(Path(output_dir) / 'test_split').as_posix()}"

        if (Path(output_dir) / f"test_results.json").exists() and \
            (Path(output_dir) / f"test_results_rand.json").exists():
            print(f"Skipping {output_dir} because it already exists")
            continue

        cmd = [
            "python3", "-m", "codetrace.scripts.launch_steer",
            "--model", f"/work/nvme/bcbj/franlucc/models/{model}",
            "--candidates", "None",
            "--output-dir", output_dir,
            "--layers", layers,
            "--steer-name", "None",
            "--test-name", "test_split",
            "--tensor-name", "steering_tensor.pt",
            "--test-size", "100",
            "--collect-all-layers", 
            "--run-steering-splits", *RUN_SPLITS,
            "--dedup-type-threshold", "-1", # prepared beforehand
            "--dedup-prog-threshold", "-1", # prepared beforehand
            "--steering-field", steering_field
        ]
        
        print(" ".join(cmd))
        if not dry_run:
            subprocess.run(cmd, check=True)

def main_with_args(
    model: str, 
    test_ds: Path,
    split: Optional[str],
    test_ds_mutations: str,
    results_label: str,
    steering_field: str,
    steering_tensor: Path,
    num_layers: int, 
    interval: int, 
    lang: str,
    results_dir: Path,
    dry_run: bool,
    multiplier: int = 1
):
    model_path = Path(f"/work/nvme/bcbj/franlucc/models/{model}")
    if not model_path.exists():
        print(f"Model {model_path} does not exist", file=sys.stderr)
        sys.exit(1)

    # load steering tensor + check steering tensor has all layers computed
    if not Path(steering_tensor).exists():
        print(f"Steering tensor {steering_tensor} does not exist", file=sys.stderr)
        sys.exit(1)

    tensor = torch.load(steering_tensor)
    for l in range(num_layers):
        assert tensor[l].sum() != 0, f"Tensor {steering_tensor} layer {l} was not collected!"
    if multiplier != 1:
        tensor = tensor * multiplier
    
    # load test split
    test_split = datasets.load_from_disk(test_ds)
    if split:
        test_split = test_split[split]

    # for each layer interval, create a subdir in results_dir 
    # and copy steering_tensor and test split to it
    test_mutations_underscored = test_ds_mutations.replace(",","_")
    for layer_range in get_ranges(num_layers, interval):
        output_dir = results_dir / f"{results_label}steering-{lang}-{test_mutations_underscored}-{layer_range.replace(',','_')}-{model}"
        if not dry_run:
            os.makedirs(output_dir) # must not exist beforehand to prevent errors
            torch.save(tensor, output_dir / "steering_tensor.pt")
            test_split.save_to_disk(output_dir / "test_split")

    # steering over layers
    _run_layer_steer(num_layers, interval, test_ds_mutations, lang, model, results_label, steering_field, dry_run)

def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--model", type=str)
    parser.add_argument("--test-ds", type=Path)
    parser.add_argument("--test-ds-mutations", type=str, help="Comma separated: [types, delete, vars]")
    parser.add_argument("--steering-field", type=str)
    parser.add_argument("--steering-tensor", type=Path)
    parser.add_argument("--results-label", type=str)
    parser.add_argument("--lang", type=str)
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--interval", type=int)

    parser.add_argument("--multiplier", type=float, default=1.)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()
    assert "-" not in args.results_label, "To use plotting script, please do not include - in labels."

    assert args.test_ds_mutations in ["vars", "types","delete","vars,delete","types,delete","types,vars",
                                      "delete,vars,types"]

    main_with_args(**vars(args))

if __name__ == "__main__":
    main()
