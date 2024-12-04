import datasets
import argparse
from pathlib import Path
import sys
import os
from huggingface_hub import upload_folder, upload_file
from tqdm import tqdm
import uuid
from shutil import copyfile, rmtree
import yaml
from typing import Dict,List,Any
from concurrent.futures import ProcessPoolExecutor


SPLIT = "train"

HELP= """
Utilities to upload and download configurations from nuprl/type-steering.

These are actions that huggingface-cli does not seem to directly support.

To delete a configuration, you can use:

    datasets-cli delete_from_hub nuprl/type-steering CONFIG_NAME
""".strip()

AUTH_TOKEN = os.environ.get("HF_AUTH_TOKEN",None)

def load_results_dataset(path: Path) -> datasets.DatasetDict:
    """
    Given a path with a test_steering_results, test_steering_results_rand
    and optional steer_steering_results subdir, create a dataset from these
    with corresponding splits test,rand,steer
    """
    steer_path = (path / "steer_steering_results")
    test_path = (path / "test_steering_results")
    rand_path = (path / "test_steering_results_rand")
    dataset_dict = {}
    cast_errors_to_str = (lambda x: {**x, "errors": "" if not x["errors"] else x["errors"]})
    
    if steer_path.exists():
        steer_split = datasets.Dataset.load_from_disk(steer_path).map(cast_errors_to_str)
        dataset_dict["steer"] = steer_split
    if test_path.exists():
        test_split = datasets.Dataset.load_from_disk(test_path).map(cast_errors_to_str)
        dataset_dict["test"] = test_split
    if rand_path.exists():
        rand_split = datasets.Dataset.load_from_disk(rand_path).map(cast_errors_to_str)
        dataset_dict["rand"] = rand_split
    
    return datasets.DatasetDict(dataset_dict)

def build_infodict(path: Path) -> dict:
    """
    {config_name: ...,
    features: [{name: ..., dtype:...}],
    splits: [{name: ..., num_examples: ...}],
    dataset_size: ...,
    data_files: [{split: ..., path: ...}]}
    """
    infodict = {"config_name": path.name, "splits":[], "data_files":[]}
    for split_name in ["rand","steer","test"]:
        split_path = (path / (split_name + "-0-of-1.parquet"))
        if split_path.exists():
            split_ds = datasets.Dataset.from_parquet(split_path.as_posix())
            infodict["splits"].append({"name": split_name, "num_examples": len(split_ds)})
            infodict["features"] = split_ds.features._to_yaml_list()
            infodict["data_files"].append({"split": split_name, "path": f"{path.name}/{split_name}-*"})
    return infodict

def process_result(subpath: Path, tempdir: Path) -> Dict[str, Any]:
    """
    Process a single result directory and return its dataset info and data files.
    """
    results_temp_path = tempdir / subpath.name
    os.makedirs(results_temp_path, exist_ok=True)
    results_dict = load_results_dataset(subpath)

    if len(results_dict) == 0:
        return None  # Skip if no results

    for split_name, dataset in results_dict.items():
        file_name = f"{split_name}-0-of-1.parquet"
        file_path = os.path.join(results_temp_path, file_name)
        dataset.to_parquet(file_path)

    infodict = build_infodict(results_temp_path)
    datafiles = {
        "config_name": infodict["config_name"],
        "data_files": infodict.pop("data_files"),
    }

    return {"infodict": infodict, "datafiles": datafiles}


def create_results_repo(path: Path, num_proc=10, pattern="steering*") -> Path:
    """
    Copy the steering results to a temporary path, formatted as a Hugging Face dataset
    repo with subsets. Use multiprocessing to speed up the operation.
    Return the temp path.
    """
    tempdir = Path(f"/tmp/{uuid.uuid4()}")
    os.makedirs(tempdir, exist_ok=True)

    results = list(path.glob(pattern))
    config_infos, config_datafiles = [], []

    with ProcessPoolExecutor(max_workers=num_proc) as executor:
        results_data = list(
            tqdm(
                executor.map(process_result, results, [tempdir] * len(results)),
                total=len(results),
                desc="Loading results data",
            )
        )

    for result in results_data:
        if result:
            config_infos.append(result["infodict"])
            config_datafiles.append(result["datafiles"])

    # Write the README with YAML dataset info
    with open(tempdir / "README.md", "w") as fp:
        yaml_str = yaml.dump({"dataset_info": config_infos, "configs": config_datafiles})
        fp.write(f"---\n{yaml_str}---")

    return tempdir


def create_vectors_repo(path: Path, pattern="steering*") -> Path:
    """
    Copy the steering vectors to a temporary path, formatted as a huggingface model
    repo with .pt files. Return the temp path.
    """
    tempdir = Path(f"/tmp/{uuid.uuid4()}")
    os.makedirs(tempdir)

    vectors = list(path.glob(pattern))
    for subpath in tqdm(vectors, total=len(vectors), desc="Loading vectors data"):
        if subpath.is_dir():
            vector_temp_path = (tempdir / f"{subpath.name}.pt")
            steering_vector = (subpath / "steering_tensor.pt")
            if steering_vector.exists():
                copyfile(steering_vector, vector_temp_path)

    return tempdir

def upload_results_folder(path: Path, create_pr: bool = False, search_pattern: str = "steering*"):
    """
    Pushes two commits: one for result datasets repo; one for steering vectors repo.
    Uploads entire directories each time.

    Use create_pr for a dry run to prevent overwriting
    """
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"]="true"
    datasets.disable_progress_bars()
    vectors_repo = create_vectors_repo(path, pattern=search_pattern)
    results_repo = create_results_repo(path, pattern=search_pattern)
    print(f"Vectors temporary path {vectors_repo}, Results temporary path {results_repo}")
    print("Pushing to hub, this may take a while...")
    upload_folder(
        repo_id="nuprl-staging/type-steering-results",
        folder_path=results_repo,
        token=AUTH_TOKEN,
        repo_type="dataset",
        create_pr=create_pr
    )
    upload_folder(
        repo_id="nuprl-staging/steering-tensors",
        folder_path=vectors_repo,
        token=AUTH_TOKEN,
        repo_type="model",
        create_pr=create_pr
    )
    print("Done pushing.")
    rmtree(vectors_repo, ignore_errors=True)
    rmtree(results_repo, ignore_errors=True)

def upload_results_file(path: Path, create_pr: bool = False):
    config_name = path.name
    results_ds = load_results_dataset(path)
    results_ds.push_to_hub(f"nuprl-staging/type-steering-results", config_name=config_name, token=AUTH_TOKEN,
                           commit_message=f"Uploading {config_name} results")
    print(results_ds)

    upload_file(
        path_or_fileobj=os.path.join(path, "steering_tensor.pt"),
        path_in_repo= config_name + ".pt",
        repo_id="nuprl-staging/steering-tensors",
        repo_type="model",
        commit_message=f"Upload {config_name}.pt tensor",
        token=AUTH_TOKEN,
        create_pr=create_pr
    )

def upload_results(path: Path, file_upload: bool, create_pr: bool, search_pattern: str):
    if file_upload:
        upload_results_file(path, create_pr=create_pr)
    else:
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"]="true"
        print(f"Search pattern: {search_pattern}")
        upload_results_folder(path, create_pr=create_pr, search_pattern=search_pattern)

def upload(path: Path):
    config_name = path.name

    the_dataset = datasets.Dataset.load_from_disk(path.as_posix())
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
    print(the_dataset)

def main():
    parser = argparse.ArgumentParser(description=HELP)
    subparsers = parser.add_subparsers(dest="command")

    parser_upload = subparsers.add_parser("upload")
    parser_upload.add_argument("path", type=Path)

    parser_upload_results = subparsers.add_parser("upload_results")
    parser_upload_results.add_argument("path", type=Path)
    parser_upload_results.add_argument("-f", "--file-upload", action="store_true")
    parser_upload_results.add_argument("-pr", "--open-pull-request", action="store_true")
    parser_upload_results.add_argument("-s", "--search-pattern", type=str, default="steering*")

    parser_download = subparsers.add_parser("download")
    parser_download.add_argument("config_name", type=str)
    parser_download.add_argument("-o", "--output_dir", default=Path("."), type=Path)

    args = parser.parse_args()

    if args.command == "upload":
        upload(args.path)
    elif args.command == "upload_results":
        print(f"Open pull request: {args.open_pull_request}")
        upload_results(args.path, args.file_upload, args.open_pull_request, args.search_pattern)
    elif args.command == "download":
        download(args.config_name, args.output_dir)
    else:
        raise ValueError(f"Unknown command: {args.command}")

if __name__ == "__main__":
    main()


