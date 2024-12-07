import datasets
from tqdm import tqdm
from multiprocessing import cpu_count
from codetrace.fast_utils import batched_apply, make_batches
import itertools as it

def model_n_layer(model: str) -> int:
    if "qwen" in model.lower():
        return 28
    elif "codellama" in model.lower():
        return 32
    elif "starcoderbase-1b" in model.lower():
        return 24
    elif "starcoderbase-7b" in model.lower():
        return 42
    elif "llama" in model.lower():
        return 24
    else:
        raise NotImplementedError(f"Model {model} model_n_layer not implemented!")

def get_ranges(num_layers: int, interval: int):
    for i in range(0, num_layers):
        if i + interval <= num_layers:
            yield "_".join(map(str, range(i, i + interval)))

def check_all_generations():
    progress_bar = tqdm(total=(2*4*7), desc="Checking generations")
    for lang in ["py","ts"]:
        for model in ["CodeLlama-7b-Instruct-hf",
                      "qwen2p5_coder_7b_base",
                      "starcoderbase-7b","starcoderbase-1b"]:
            for mut in ["types","vars","delete","vars_delete","types_delete","types_vars","delete_vars_types"]:
                ds = datasets.load_dataset("nuprl-staging/type-steering",f"mutations-{lang}-{mut}-{model}",split="train")
                df = ds.to_pandas()
                counts_mut = df.value_counts("mutated_generated_text")
                counts = df.value_counts("generated_text")
                if counts_mut.get("", 0) > 0:
                    print(f"mutations: {lang}-{mut}-{model}: {counts_mut['']}")
                if counts.get("", 0) > 0:
                    print(f"generated: {lang}-{mut}-{model}: {counts['']}")
                progress_bar.update(1)
    progress_bar.close()

ALL_MODELS = ["CodeLlama-7b-Instruct-hf","qwen2p5_coder_7b_base",
              "Llama-3.2-3B-Instruct",
              "starcoderbase-7b","starcoderbase-1b"]
def check_all_generations_in_test(
    keydict: list[dict], 
    split="test", 
    disable_tqdm=False
) -> list[str]:
    assert not keydict or len(keydict) == 1
    if len(keydict) == 1:
        keydict = keydict[0]
    langs=keydict.pop("langs",["py","ts"])
    all_models=keydict.pop("models",ALL_MODELS)
    failed = []
    progress_bar = tqdm(total=(len(langs)*len(all_models)*7), desc=f"Checking generations in {split}", disable=disable_tqdm)
    for lang in langs:
        for model in all_models:
            for mut in ["types","vars","delete","vars_delete","types_delete","types_vars","delete_vars_types"]:
                for interval in [1,3,5]:
                    for layers in get_ranges(model_n_layer(model), interval):
                        name=f"steering-{lang}-{mut}-{layers}-{model}"
                        try:
                            ds = datasets.load_dataset("nuprl-staging/type-steering-results",name=name,
                                                    split=split,trust_remote_code=True)
                        except ValueError as e:
                            print(name)
                            continue
                        
                        df = ds.to_pandas()
                        counts_mut = df.value_counts("mutated_generated_text")
                        counts = df.value_counts("generated_text")
                        if counts_mut.get("", 0) > 0:
                            failed.append(f"mutations {split}: {lang}-{mut}-{layers}-{model}: {counts_mut['']}")
                        if counts.get("", 0) > 0:
                            failed.append(f"generated {split}: {lang}-{mut}-{layers}-{model}: {counts['']}")
                        failed.append(f"Counts of null predictions: {counts.get('', 0)} {counts_mut.get('', 0)}")
                progress_bar.update(1)
    progress_bar.close()
    return failed

def multiproc_check_results(split):
    keys = [{"langs": [l], "models": [m]} for m,l in it.product(ALL_MODELS, ["py","ts"])]
    batches = make_batches(keys, len(keys))
    results = batched_apply(batches, len(keys), check_all_generations_in_test, split=split, disable_tqdm=True)
    print("\n".join(results))

if __name__ == "__main__":
    datasets.disable_progress_bars()
    multiproc_check_results("test")
    multiproc_check_results("steer")
    check_all_generations()
    
