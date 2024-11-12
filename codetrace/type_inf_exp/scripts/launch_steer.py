from codetrace.type_inf_exp.steering import SteeringManager
from argparse import ArgumentParser
from nnsight import LanguageModel
from typing import List,Dict
import json
import os
from shutil import rmtree

def evaluate(results_ds) -> Dict:
    df = results_ds.to_pandas()
    df["steer_success"] = df["steered_predictions"] == df["fim_type"]
    return {
        "num_succ": df["steer_success"].sum(),
        "total": df["steer_success"].count(),
        "mean_succ": df["steer_success"].mean()
    }

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
    test_size:int
):
    model = LanguageModel(model, torch_dtype=dtype,device_map="cuda")
    smanager = SteeringManager(
        model,
        candidates,
        output_dir,
        steer_name,
        test_name,
        tensor_name,
        max_num_candidates
    )
    # 1. make splits
    steer_split, test_split = smanager.steer_test_splits(test_size, 3, 25)
    print("Steer split:\n",steer_split,"Test_split:\n", test_split)
    smanager.save_data(steer_split, steer_name)
    smanager.save_data(test_split, test_name)

    # 2. make tensor
    steering_tensor = smanager.create_steering_tensor(collect_batchsize)
    smanager.save_tensor(steering_tensor, tensor_name)

    # 3. run steering on test
    results_ds = smanager.steer("test", layers, patch_batchsize )

    # 4. analyze and save results
    smanager.save_data(results_ds, "test_steering_results")
    evaluation = evaluate(results_ds)
    print(evaluation)
    with open(os.path.join(output_dir, "test_eval.json")) as fp:
        json.dumps(evaluation, fp, indent=4)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model",type=str, required=True)
    parser.add_argument("--candidates", type=str,required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--layers", nargs="+", type=int, required=True)

    parser.add_argument("--collect-batchsize", "-b1",type=int, default=4)
    parser.add_argument("--patch-batchsize", "-b2",type=int, default=2)
    parser.add_argument("--dtype", choices=["bfloat16","float32"],default="bfloat16")
    parser.add_argument("--max-num-candidates","-n",type=int, default=2000)
    parser.add_argument("--test-size", type=int,default=500)

    parser.add_argument("--overwrite", action="store_true")
    # naming
    parser.add_argument("--steer-name", required=True)
    parser.add_argument("--test-name", required=True)
    parser.add_argument("--tensor-name", required=True)

    args = parser.parse_args().__dict__
    if args.pop("overwrite", None) and os.path.exists(args["output_dir"]):
        rmtree(args["output_dir"])
    main(**args)