from pathlib import Path
import argparse
import sys

DESCRIPTION = """
Prints sbatch commands to perform activation steering for a given model and
language. It looks for mutations named mutations-LANG-MUTATIONS-MODEL in
the results directory and prints commands to steer intervals of 1,3, and 5
layers. On the command line, you need to specify the total number of layers
in the model.
""".strip()

# sbatch codetrace/bin/steering_delta.py --model qwen2p5_coder_7b_base --mutations delete --lang py --num-layers 28 --interval 1
def main_with_args(model: str, lang: str, num_layers: int, interval: int, results_dir: Path):
    model_path = Path(f"/work/nvme/bcbj/models/{model}")
    if not model_path.exists():
        print(f"Model {model} does not exist", file=sys.stderr)
        sys.exit(1)

    mutations_paths = list(results_dir.glob(f"mutations-{lang}-*-{model}"))
    if len(mutations_paths) == 0:
        print(f"No mutations found for {lang} and {model}", file=sys.stderr)
        sys.exit(1)

    for p in mutations_paths:
        mutations = p.name.split("-")[-2].replace("_", ",")
        print(f"sbatch codetrace/bin/steering_delta.sbatch --model {model} --mutations {mutations} --lang {lang} --num-layers {num_layers} --interval {interval}")

def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--model", type=str)
    parser.add_argument("--lang", type=str)
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--interval", type=int)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    main_with_args(args.model, args.lang, args.num_layers, args.interval, args.results_dir)

if __name__ == "__main__":
    main()
