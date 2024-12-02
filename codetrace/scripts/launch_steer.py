from codetrace.steering import SteeringManager
from codetrace.utils import load_dataset
from argparse import ArgumentParser
from nnsight import LanguageModel
from typing import List,Dict,Optional
import json
import os
from shutil import rmtree

def evaluate(results_ds) -> Dict:
    df = results_ds.to_pandas()
    df["steer_success"] = df["steered_predictions"] == df["fim_type"]
    return {
        "num_succ": float(df["steer_success"].sum()),
        "total": float(df["steer_success"].count()),
        "mean_succ": float(df["steer_success"].mean())
    }

def run_steer(
    smanager:SteeringManager,
    split_name:str,
    layers:List[int],
    patch_batchsize:int,
    do_random_ablation:bool
):
    results_ds = smanager.steer(split_name, layers, patch_batchsize, do_random_ablation=do_random_ablation)
    suffix = "_rand" if do_random_ablation else ""

    # 4. analyze and save results
    smanager.save_data(results_ds, f"{split_name}_steering_results{suffix}")
    evaluation = evaluate(results_ds)
    print(evaluation)
    with open(os.path.join(smanager.cache_dir, f"{split_name}_results{suffix}.json"),"w") as fp:
        json.dump(evaluation, fp, indent=3)

def main(
    model:str,
    dtype:str,
    candidates:str,
    layers:List[int],
    output_dir:str,
    steer_name:str,
    test_name:str,
    tensor_name:str,
    collect_batchsize:int,
    patch_batchsize:int,
    max_num_candidates:int,
    test_size:int,
    subset:str,
    split:Optional[str],
    run_steering_splits: Optional[List[str]] = None,
    collect_all_layers: bool = False
):
    candidates = load_dataset(candidates, split=split,name=subset)
    model = LanguageModel(model, torch_dtype=dtype,device_map="cuda",dispatch=True)
    smanager = SteeringManager(
        model,
        candidates,
        output_dir,
        steer_name,
        test_name,
        tensor_name,
        max_num_candidates,
        only_collect_layers=None if collect_all_layers else layers
    )
    # 1. make splits
    steer_split, test_split = smanager.steer_test_splits(test_size, 3, 25)
    print("Steer split:\n",steer_split,"Test_split:\n", test_split)
    smanager.save_data(steer_split, steer_name)
    smanager.save_data(test_split, test_name)

    # 2. make tensor
    steering_tensor = smanager.create_steering_tensor(collect_batchsize)
    smanager.save_tensor(steering_tensor, tensor_name)

    # check valid options
    assert not run_steering_splits or set(run_steering_splits).issubset(set(["test","steer","rand"]))
    
    # 3. run steering on test
    if not run_steering_splits or "test" in run_steering_splits:
        run_steer(smanager, "test", layers, patch_batchsize, False)

    # 4. run steering on test with random tensor
    if not run_steering_splits or "rand" in run_steering_splits:
        run_steer(smanager, "test", layers, patch_batchsize, True)

    # 5. run steering on steer
    if not run_steering_splits or "steer" in run_steering_splits: 
        run_steer(smanager, "steer", layers, patch_batchsize, False)

    smanager.clear_cache()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model",type=str, required=True)
    parser.add_argument("--candidates", type=str,required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--layers", type=str, required=True)
    # dataset
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--subset", type=str, default=None)

    # naming
    parser.add_argument("--steer-name", required=True)
    parser.add_argument("--test-name", required=True)
    parser.add_argument("--tensor-name", required=True)

    parser.add_argument("--collect-batchsize", "-b1",type=int, default=4)
    parser.add_argument("--patch-batchsize", "-b2",type=int, default=2)
    parser.add_argument("--dtype", choices=["bfloat16","float32"],default="bfloat16")
    parser.add_argument("--max-num-candidates","-n",type=int, default=3000)
    parser.add_argument("--test-size", type=int,default=100)

    parser.add_argument("--run-steering-splits", type=str, nargs="+", choices=["test","steer","rand"], default=None)
    parser.add_argument("--collect-all-layers", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args().__dict__
    if args.pop("overwrite", None) and os.path.exists(args["output_dir"]):
        rmtree(args["output_dir"])
    args["layers"] = [int(l.strip()) for l in args["layers"].split(',') if l != ""]
    print(f"Layers: {args['layers']}")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main(**args)