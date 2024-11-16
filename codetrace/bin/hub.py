import datasets
import argparse
from pathlib import Path
import sys

SPLIT = "train"

HELP= """
Utilities to upload and download configurations from nuprl/type-steering.

These are actions that huggingface-cli does not seem to directly support.

To delete a configuration, you can use:

    datasets-cli delete_from_hub nuprl/type-steering CONFIG_NAME
""".strip()


def upload(path: Path):
    config_name = path.name

    the_dataset = datasets.Dataset.load_from_disk(path)
    the_dataset.push_to_hub(f"nuprl/type-steering", config_name=config_name, split=SPLIT)

def download(config_name: str, output_dir: Path):
    output_dir = output_dir / config_name

    if output_dir.is_file():
        print(f"Output directory {output_dir} is a file")
        sys.exit(1)
    
    # check that the path exists and is an empty directory
    if output_dir.is_dir() and any(output_dir.iterdir()):
        print(f"Output directory {output_dir} is not empty")
        sys.exit(1)

    the_dataset = datasets.load_dataset("nuprl/type-steering", name=config_name, split=SPLIT)
    the_dataset.save_to_disk(output_dir)

def main():
    parser = argparse.ArgumentParser(description=HELP)
    subparsers = parser.add_subparsers(dest="command")

    parser_upload = subparsers.add_parser("upload")
    parser_upload.add_argument("path", type=Path)

    parser_download = subparsers.add_parser("download")
    parser_download.add_argument("config_name", type=str)
    parser_download.add_argument("-o", "--output_dir", default=Path("."), type=Path)

    args = parser.parse_args()

    if args.command == "upload":
        upload(args.path)
    elif args.command == "download":
        download(args.config_name, args.output_dir)
    else:
        raise ValueError(f"Unknown command: {args.command}")

if __name__ == "__main__":
    main()


