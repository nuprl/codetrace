from codetrace.steering import SteeringManager
from codetrace.utils import load_dataset, print_color
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
    do_random_ablation:bool,
    steering_field:Optional[str] = None
):
    results_ds = smanager.steer(split_name, layers, patch_batchsize, 
                                do_random_ablation=do_random_ablation,
                                steering_field=steering_field)
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
    candidates:Optional[str],
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
    run_steering_splits: List[str] = ["test","rand","steer"],
    collect_all_layers: bool = False,
    dedup_prog_threshold: int = 25,
    dedup_type_threshold: int = 4,
    steering_field: Optional[str] = None
):
    if candidates:
        candidates = load_dataset(candidates, split=split,name=subset)
    
    model = LanguageModel(model, torch_dtype=dtype,device_map="cuda",dispatch=True)
    smanager = SteeringManager(
        model,
        output_dir,
        candidates,
        steer_name,
        test_name,
        tensor_name,
        max_num_candidates,
        only_collect_layers=None if collect_all_layers else layers
    )

    # 1. make splits if they do not exist and save
    steer_split, test_split = smanager.steer_test_splits(test_size, dedup_prog_threshold, dedup_type_threshold)
    print("Steer split:\n",steer_split,"Test_split:\n", test_split)

    smanager.save_data(test_split, test_name)
    if "steer" in run_steering_splits:
        smanager.save_data(steer_split, steer_name)

    # 2. make tensor if it does not exist
    steering_tensor = smanager.create_steering_tensor(collect_batchsize)
    smanager.save_tensor(steering_tensor, tensor_name)

    # 3. run steering on splits
    if "test" in run_steering_splits:
        print_color("[TEST SPLIT]", "green")
        run_steer(smanager, "test", layers, patch_batchsize, False, steering_field)

    if "rand" in run_steering_splits:
        print_color("[RAND SPLIT]", "red")
        run_steer(smanager, "test", layers, patch_batchsize, True, steering_field)

    if "steer" in run_steering_splits:
        print_color("[STEER SPLIT]", "yellow")
        run_steer(smanager, "steer", layers, patch_batchsize, False, steering_field)

    smanager.clear_cache()

def none_or_str(value):
    if value == 'None':
        return None
    return value

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--model",type=str, required=True)
    parser.add_argument("--candidates", type=none_or_str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--layers", type=str, required=True)
    # dataset
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--subset", type=str, default=None)
    parser.add_argument("--dedup-prog-threshold", type=int, default=25)
    parser.add_argument("--dedup-type-threshold", type=int, default=4)
    # naming
    parser.add_argument("--steer-name", required=True)
    parser.add_argument("--test-name", required=True)
    parser.add_argument("--tensor-name", required=True)
    parser.add_argument("--steering-field", type=str, default=None)

    parser.add_argument("-b1","--collect-batchsize", type=int, default=4)
    parser.add_argument("-b2","--patch-batchsize",type=int, default=2)
    parser.add_argument("--dtype", choices=["bfloat16","float32"],default="bfloat16")
    parser.add_argument("--max-num-candidates","-n",type=int, default=3000)
    parser.add_argument("--test-size", type=int,default=100)

    parser.add_argument("--run-steering-splits", type=str, nargs="+", choices=["test","steer","rand"], 
                        default=["test","rand","steer"])
    parser.add_argument("--collect-all-layers", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()
    # check valid options
    assert set(args.run_steering_splits).issubset(set(["test","steer","rand"]))
    args = vars(args)
    
    if args.pop("overwrite", None) and os.path.exists(args["output_dir"]):
        rmtree(args["output_dir"])
    args["layers"] = [int(l.strip()) for l in args["layers"].split(',') if l != ""]
    print(f"Layers: {args['layers']}")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main(**args)