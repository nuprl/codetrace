from codetrace.type_inf_exp.request_patch import *
from codetrace.utils import *
from einops import rearrange
from argparse import ArgumentParser, Namespace
from collections import Counter
from baukit.nethook import TraceDict
from transformers import AutoModelForCausalLM, AutoTokenizer
from copy import deepcopy
import json
import gzip

def fit_test_split(dataset : datasets.Dataset, args):
    correct = dataset.remove_columns(["renamed_prompt","renamed_variables","renamed_percent","correct"]).rename_column("original_prompt","prompt")
    print(correct)
    incorrect = dataset.remove_columns(["original_prompt","correct"]).rename_column("renamed_prompt","prompt")
    print(incorrect)
    
    if args.test_size > 0:
        # set aside some incorrect prompts
        random.seed(4)
        hexshas = list(incorrect["name"])
        hexshas = random.sample(hexshas, int(len(hexshas) * args.test_size))
        incorrect_eval = incorrect.filter(lambda x : x["name"] in hexshas)
        incorrect = incorrect.filter(lambda x : x["name"] not in hexshas)
        correct = correct.filter(lambda x : x["name"] not in hexshas)
        return correct, incorrect, incorrect_eval
    else:
        return correct, incorrect, None

def _pretty_print(ds) -> str:
    """
    Give some information about how balanced the ds is
    """
    df = pd.DataFrame(ds)
    s = ""
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        s += str(df["name"].value_counts())
        s += "\n"
        s += str(len(df))
    return s

def _truncate_at_stop_tokens(tokens: List[int],
                             stop_tokens : List[str],
                            tokenizer : AutoTokenizer) -> List[int]:
    """
    Truncate generated at stop tokens AFTER comments
    """
    def find_earliest_occurrence(n, subsets):
        for subset in subsets:
            subset_tuple = tuple(subset)
            
            for i in range(len(n) - len(subset) + 1):
                if tuple(n[i:i+len(subset)]) == subset_tuple:
                    return i
        return len(n)

    stop_tokens = [tokenizer.encode(x) for x in stop_tokens]
    # find earliest index of a stop token in tokens
    idx = find_earliest_occurrence(tokens, stop_tokens)
    return tokens[:idx]
        
            
        

def get_layername(layers : List[int]) -> List[str]:
    """
    Get layer names from layer numbers
    """
    return [f"transformer.h.{l}" for l in layers]

def get_layeridx(layer: str) -> int:
    """
    Get layer numbers from layer names
    """
    return int(layer.split(".")[-1])

def steer(
    hf_model,
    tokenizer,
    prompts : Union[List[str], str],
    patch_tensor : torch.Tensor,
    layers_to_patch : Union[List[int],int],
    tokens_to_patch : Union[List[int],int],
    patch_mode : str,
    batch_size : int = 1,
    max_out : int = 512
):
    """
    Need to steer with generation
    TODO: 
    - implement batching
    - implement patch_mode
    - implement patch token str and not int
    """
    layers_to_patch, tokens_to_patch, prompts = arg_to_list(layers_to_patch), arg_to_list(tokens_to_patch), arg_to_list(prompts)
    assert isinstance(tokens_to_patch[0],int)
    assert patch_mode in ["add"]
    assert len(prompts) == batch_size == 1, "Batching not implemented yet"
    
    patch_tensor = patch_tensor.to(hf_model.device)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    tokens_to_patch = [arg_to_literal(x, n=len(tokenizer.tokenize(prompts[0]))) for x in tokens_to_patch]


    input_ids = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).input_ids.to("cuda")
    layers_to_patch = get_layername(layers_to_patch)
    
    def edit_fn(x, layer):
        # tuple is (out, ?)
        # shape should be [n_prompts, n_tokens, n_embd]
        if layer in layers_to_patch:
            output = x[0]
            output[:,tokens_to_patch,:] += patch_tensor[get_layeridx(layer),:,:]
            return (output, None)
        else:
            return x
    
    generated = []
    with TraceDict(hf_model, 
                   layers=get_layername(list(range(hf_model.config.n_layer))),
                   edit_output=edit_fn
                   ) as ret:
        for i in range(max_out):
            td = hf_model(input_ids)
            out_tok = td.logits[:,-1,:].softmax(-1).argmax(-1)
            generated.append(out_tok)
            input_ids = torch.cat([input_ids, out_tok.unsqueeze(1)], dim=1)
    
    return generated

def main():
    # ==========================================================================================
    # PART 0: setup
    # ==========================================================================================
    steering_args = os.path.join(os.path.dirname(__file__), "args_steering.json")
    with open(steering_args, "r") as f:
        args = json.load(f)
    args = Namespace(**args)

    exp_dir = "/home/franlucc/projects/codetrace/codetrace/code_gen_exp"
    ds = datasets.load_dataset(args.dataset, split="train")

    model = LanguageModel(args.model, device_map="cuda")

    out_dir = f"{exp_dir}/exp_data/{args.outdir}"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # ==========================================================================================
    # PART 1: filter
    # ==========================================================================================
    print("...Generating fit test split...")

    correct, incorrect, incorrect_eval = fit_test_split(ds, args)

    info_incorrect = _pretty_print(incorrect)
    info_correct = _pretty_print(correct)
    if incorrect_eval is not None:
        info_eval = _pretty_print(incorrect_eval)
    else:
        info_eval = "No eval set"

    info = f"Correct\n{info_correct}\n\nIncorrect\n{info_incorrect}\n\nIncorrect Eval\n{info_eval}\n"

    with open(f"{out_dir}/data_readme.md", "w") as f:
        f.write(info)
    
    print(info)
    # ==========================================================================================
    # PART 2: averages 
    #  TODO: different ways we can extract avg
    # - last token [current]
    # - last token in FIM formulation
    # - all tokens
    # ==========================================================================================
    print(f"...Getting averages for correct and incorrect prompts...")
    
    # load steering tensor if it exists, else create it
    if os.path.exists(f"{out_dir}/steering_tensor.pt"):
        print(f"...Loading steering tensor from {out_dir}/steering_tensor.pt...")
        diff_tensor = torch.load(f"{out_dir}/steering_tensor.pt")
    else:
        print(f"...Creating steering tensor...")
        correct_prompts = correct["prompt"]
        incorrect_prompts = incorrect["prompt"]
        correct_avg_tensor = batched_get_averages(model, 
                                                correct_prompts,
                                                tokens=[-1],
                                                batch_size=args.batch_size)

        incorrect_avg_tensor = batched_get_averages(model,
                                                    incorrect_prompts,
                                                    tokens=[-1],
                                                    batch_size=args.batch_size)
        
        print(f"Correct avg tensor shape: {correct_avg_tensor.shape}")
        print(f"Incorrect avg tensor shape: {incorrect_avg_tensor.shape}")

        diff_tensor = correct_avg_tensor - incorrect_avg_tensor
        print(f"Diff tensor shape: {diff_tensor.shape}")
        diff_tensor = rearrange(diff_tensor, "l t d -> l 1 t d") # [n_layers, n_prompts, n_tokens, n_embd]

        print(f"Diff tensor shape after transform: {diff_tensor.shape}")

        torch.save(diff_tensor, f"{out_dir}/steering_tensor.pt")

    #==========================================================================================
    # Part 3: steered generations
    #==========================================================================================
    
    print(f"...Applying patch to incorrect prompts...")
    incorrect = datasets.Dataset.from_pandas(pd.DataFrame(incorrect))
    del(model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    generations = []
    for i,ex in tqdm(enumerate(incorrect), desc="Applying patches"):
        generated_idx = steer(model, 
                            tokenizer,
                            ex["prompt"],
                            diff_tensor,
                            args.layers_to_patch,
                            args.tokens_to_patch,
                            args.patch_mode)
        generated_idx = _truncate_at_stop_tokens(generated_idx, ex["stop_tokens"], tokenizer)
        generated = "".join([tokenizer.decode(x) for x in generated_idx])
        generations.append({"steered_generation" : generated, **ex})
        if i % 3 == 0:
            steering_df = pd.DataFrame(generations)
            steering_df.to_csv(f"{out_dir}/steering_results.csv")
            
    steering_df = pd.DataFrame(generations)
    
    # save into MultiPLE completions format
    cut_steering_df = steering_df[["name", "steered_generation","results", "prompt"]]
    cut_steering_df.to_csv(f"{out_dir}/steering_results.csv")
    
    steering_ds = datasets.Dataset.from_pandas(steering_df)
    steering_ds = steering_ds.rename_columns({"results":"old_results"})
    steering_ds = steering_ds.map(lambda x : {"completions" : [x["steered_generation"]], 
                                              "temperature": 0.0,
                                              'language': 'py',
                                              'top_p': 0.95,
                                              'max_tokens': 512,
                                              })
    # save as gzip of json
    json_list = steering_ds.to_list()
    dirout = f"{out_dir}/steering_completions"
    os.makedirs(dirout, exist_ok=True)
    for ex in json_list:
        with gzip.open(f"{dirout}/{ex['name']}.json.gz", "wt") as f:
            json.dump(ex, f)
    
    # save args
    with open(f"{out_dir}/eval_readme.md", "w") as f:
        # write arguments of parser
        f.write(f"\n## Arguments\n")
        parser = vars(args)
        for k,v in parser.items():
            f.write(f"{k} : {v}\n")
            
    # # ==========================================================================================
    # # PART 4: steering generation ood
    # # ==========================================================================================

    
if __name__ == "__main__":
    main()