from pathlib import Path
import argparse
import sys
import os
import shutil
import torch

DESCRIPTION = """
Prints sbatch commands to perform activation steering for a given model, language
and steering tensor. It copies the steering tensor to the result output directory
of each layer for each interval passed (1,3 or 5). It checks that the passed
steering tensor has collected all layers, in case only-collect-layers flag
was used when constructing the steering tensor.

On the command line, you need to specify the total number of layers
in the model. You also need to pass a label for the steering output directory,
as well as the steering field for the candidate dataset, to keep track of what
the steering experiment is. If dry-run is passed, it will only print out
sbatch commands and not copy the tensors. If no dry-run, code will throw an error
if the result output directory already exists; this is to prevent human errors.
""".strip()

def get_ranges(num_layers: int, interval: int):
    for i in range(0, num_layers):
        if i + interval <= num_layers:
            yield ",".join(map(str, range(i, i + interval)))

def main_with_args(
    model: str, 
    candidate_ds: Path,
    results_label: str,
    steering_field: str,
    steering_tensor: Path,
    num_layers: int, 
    interval: int, 
    lang: str,
    results_dir: Path,
    dry_run: bool
):
    model_path = Path(f"/work/nvme/bcbj/franlucc/models/{model}")
    if not model_path.exists():
        print(f"Model {model} does not exist", file=sys.stderr)
        sys.exit(1)

    # check steering tensor has all layers computer
    tensor = torch.load(steering_tensor)
    for l in range(num_layers):
        assert tensor[l].sum() != 0, f"Tensor {steering_tensor} layer {l} was not collected!"
    
    # for each layer interval, create a subdir in results_dir 
    # and copy steering_tensor to it
    for layer_range in get_ranges(num_layers, interval):
        output_dir = results_dir / f"precomputed_steering-{lang}-{results_label}-{layer_range.replace(',','_')}-{model}"
        if not dry_run:
            os.makedirs(output_dir)
            shutil.copyfile(steering_tensor, output_dir / "steering_tensor.pt")
        print(f"sbatch codetrace/bin/steer_with_precomputed_tensor.sbatch {model} {candidate_ds} {output_dir} {layer_range} {steering_field}")

def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--model", type=str)
    parser.add_argument("--candidate-ds", type=Path)
    parser.add_argument("--steering-field", type=str)
    parser.add_argument("--steering-tensor", type=Path)
    parser.add_argument("--results-label", type=str)
    parser.add_argument("--lang", type=str)
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--interval", type=int)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    assert "-" not in args.results_label, "To use plotting script, please do not include - in labels."

    main_with_args(args.model, Path(args.candidate_ds), args.results_label,
                   args.steering_field, Path(args.steering_tensor), 
                   args.num_layers, args.interval, 
                   args.lang, args.results_dir, args.dry_run)

if __name__ == "__main__":
    main()
