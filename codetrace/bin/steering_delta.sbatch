#!/work/nvme/bcbj/aguha/venv-x86/bin/python3
#SBATCH --job-name=typesteering_steering
#SBATCH --partition=gpuA40x4
#SBATCH --account=bcbj-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=60G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

# You must run this script in the codetrace directory and in an environment
# in which it works.

import argparse
import subprocess
from pathlib import Path
def get_ranges(num_layers: int, interval: int):
    for i in range(0, num_layers):
        if i + interval <= num_layers:
            yield ",".join(map(str, range(i, i + interval)))


def main_with_args(model: str, mutations: str, lang: str, num_layers: int, interval: int, dry_run: bool):
    for layers in get_ranges(num_layers, interval):
        mutation_underscored = mutations.replace(",", "_")
        layers_underscored = layers.replace(",", "_")
        output_dir = f"results/steering-{lang}-{mutation_underscored}-{layers_underscored}-{model}"

        if (Path(output_dir) / "test_results.json").exists() and (Path(output_dir) / "steer_results.json").exists() and (Path(output_dir) / "test_results_rand.json").exists():
            print(f"Skipping {output_dir} because it already exists")
            continue

        cmd = [
            "python3", "-m", "codetrace.scripts.launch_steer",
            "--model", f"/work/nvme/bcbj/models/{model}",
            "--candidates", f"results/mutations-{lang}-{mutation_underscored}-{model}",
            "--output-dir", output_dir,
            "--layers", layers,
            "--steer-name", "steering_split",
            "--test-name", "test_split",
            "--tensor-name", "steering_tensor.pt",
            "-n", "-1",
            "--test-size", "100",
        ]
        print(" ".join(cmd))
        if not dry_run:
            subprocess.run(cmd, check=True)


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
