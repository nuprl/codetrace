from argparse import Namespace, ArgumentParser
from codetrace.type_inf_exp.scripts.launch_steer import main as main_steer
import datasets
import torch

def pipeline(args):
    if args.evaldir == None:
        args.evaldir = args.datadir
    
    args_data = {
        "datadir": args.datadir,
        "source_dataset": args.source_dataset,
        "max_size": -1,
        "shuffle": True,
        # "do_fit_matching_pairs": True,
        "dedup_type_threshold": 25,
        "dedup_prog_threshold": 3,
        "test_size": 0.2,
        "model": args.model,
        "action": "make_steering_data_splits",
        "seed": args.seed
    }

    args_tensor = {
        "datadir": args.datadir,
        "tokens_to_patch": ["<fim_middle>"],
        "batch_size": args.batchsize,
        "max_size": args.max_size, 
        "shuffle": True,
        "model": args.model,
        "steering_tensor_name": args.tensor_name,
        "action": "make_steering_tensor",
        "seed": args.seed
    }

    args_eval ={
        "steering_tensor": f"{args.datadir}/{args.tensor_name}",
        "model": args.model,
        "expdir": args.expdir,
        "evaldir": args.evaldir,
        "batch_size": args.batchsize,
        "max_size": args.max_size, 
        "shuffle": True,
        "patch_mode": "add",
        "tokens_to_patch": ["<fim_middle>"],
        "layers_to_patch": [10,11,12,13,14],
        "custom_decoder": False,
        "multiplier": False,
        "action": "run_steering",
        "seed": args.seed
    }

    args_data = Namespace(**args_data)
    args_tensor = Namespace(**args_tensor)
    args_eval = Namespace(**args_eval)

    main_steer(args_data)
    main_steer(args_tensor)
    main_steer(args_eval)

    if args.rand_steering_tensor != None:
        print("[RAND] Making a steering tensor of random values")
        if "starcoderbase-1b" in args.model:
            rand_tensor = torch.rand((24,1,1,2048))
        elif "starcoderbase-7b" in args.model:
            rand_tensor = torch.rand((42,1,1,4096))
        else:
            raise ValueError("args.model not supported for random tensor")
        
        torch.save(rand_tensor, f"{args.datadir}/random_steering_tensor.pt")
        
        args_rand_eval ={
            "steering_tensor": f"{args.datadir}/random_steering_tensor.pt",
            "model": args.model,
            "expdir": args.expdir + "_rand",
            "evaldir": args.evaldir,
            "batch_size": args.batchsize,
            "max_size": args.max_size, 
            "shuffle": True,
            "patch_mode": "add",
            "tokens_to_patch": ["<fim_middle>"],
            "layers_to_patch": [10,11,12,13,14],
            "custom_decoder": False,
            "rotation_matrix": False,
            "multiplier": False,
            "action": "run_steering",
            "seed": args.seed
        }
        args_rand_eval = Namespace(**args_rand_eval)
        main_steer(args_rand_eval)
        

if __name__=="__main__":
    datasets.disable_caching()
    print("Caching enabled?:", datasets.is_caching_enabled())
    parser = ArgumentParser()
    parser.add_argument("--datadir", type=str, required=True)
    parser.add_argument("--source_dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tensor_name", type=str, required=True)
    parser.add_argument("--expdir", type=str, required=True)
    parser.add_argument("--max_size", type=int, required=True)
    parser.add_argument("--batchsize", type=int, required=True)
    parser.add_argument("--seed", type=int, default=None) # default 42
    
    parser.add_argument("--evaldir", type=str, required=False, default=None)
    parser.add_argument("--rand_steering_tensor", action="store_true", default=None)
    args = parser.parse_args()
    print(args)
    pipeline(args)