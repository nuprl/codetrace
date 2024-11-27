import datasets
import argparse
from pathlib import Path
import sys
import os
from huggingface_hub import upload_file

SPLIT = "train"

HELP= """
Utilities to upload and download configurations from nuprl/type-steering.

These are actions that huggingface-cli does not seem to directly support.

To delete a configuration, you can use:

    datasets-cli delete_from_hub nuprl/type-steering CONFIG_NAME
""".strip()

AUTH_TOKEN = os.environ.get("HF_AUTH_TOKEN",None)

def upload_results(path: Path):
    config_name = path.name
    
    # we need to map error column to str because splits need to have same types,
    # since some splits always typecheck the error column is Null
    steer_results = datasets.Dataset.load_from_disk(os.path.join(path, "steer_steering_results")
            ).map(lambda x: {**x, "errors": "" if not x["errors"] else x["errors"]})
    test_results = datasets.Dataset.load_from_disk(os.path.join(path, "test_steering_results")
            ).map(lambda x: {**x, "errors": "" if not x["errors"] else x["errors"]})
    test_results_rand = datasets.Dataset.load_from_disk(os.path.join(path, "test_steering_results_rand")
            ).map(lambda x: {**x, "errors": "" if not x["errors"] else x["errors"]})
    
    results_ds = datasets.DatasetDict({
        "test": test_results,
        "steer": steer_results,
        "test_rand": test_results_rand
    })
    results_ds.push_to_hub(f"nuprl-staging/type-steering", config_name=config_name, token=AUTH_TOKEN,
                           commit_message=f"Uploading {config_name} results")
    print(results_ds)

    upload_file(
        path_or_fileobj=os.path.join(path, "steering_tensor.pt"),
        path_in_repo= config_name + ".pt",
        repo_id="nuprl-staging/steering-tensors",
        repo_type="model",
        commit_message=f"Upload {config_name}.pt tensor",
        token=AUTH_TOKEN
    )
    
def upload(path: Path):
    config_name = path.name

    the_dataset = datasets.Dataset.load_from_disk(path)
    the_dataset.push_to_hub(f"nuprl-staging/type-steering", config_name=config_name, split=SPLIT,
                            token=AUTH_TOKEN, commit_message=f"Uploading {config_name} data")
    print(the_dataset)

def download(config_name: str, output_dir: Path):
    output_dir = output_dir / config_name

    if output_dir.is_file():
        print(f"Output directory {output_dir} is a file")
        sys.exit(1)
    
    # check that the path exists and is an empty directory
    if output_dir.is_dir() and any(output_dir.iterdir()):
        print(f"Output directory {output_dir} is not empty")
        sys.exit(1)

    the_dataset = datasets.load_dataset("nuprl-staging/type-steering", name=config_name, split=SPLIT,)
    the_dataset.save_to_disk(output_dir)

def main():
    parser = argparse.ArgumentParser(description=HELP)
    subparsers = parser.add_subparsers(dest="command")

    parser_upload = subparsers.add_parser("upload")
    parser_upload.add_argument("path", type=Path)

    parser_upload_results = subparsers.add_parser("upload_results")
    parser_upload_results.add_argument("path", type=Path)

    parser_download = subparsers.add_parser("download")
    parser_download.add_argument("config_name", type=str)
    parser_download.add_argument("-o", "--output_dir", default=Path("."), type=Path)

    args = parser.parse_args()

    if args.command == "upload":
        upload(args.path)
    elif args.command == "upload_results":
        upload_results(args.path)
    elif args.command == "download":
        download(args.config_name, args.output_dir)
    else:
        raise ValueError(f"Unknown command: {args.command}")

if __name__ == "__main__":
    main()


