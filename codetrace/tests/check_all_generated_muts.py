import datasets
from tqdm import tqdm

def model_n_layer(model: str) -> int:
    if "qwen" in model.lower():
        return 28
    elif "codellama" in model.lower():
        return 32
    elif "starcoderbase-1b" in model.lower():
        return 24
    elif "starcoderbase-7b" in model.lower():
        return 42
    else:
        print(f"Model {model} model_n_layer not implemented!")
        return None

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

def check_all_generations_in_test(split="test"):
    progress_bar = tqdm(total=(2*4*7), desc=f"Checking generations in {split}")
    for lang in ["py","ts"]:
        for model in ["CodeLlama-7b-Instruct-hf",
                      "qwen2p5_coder_7b_base",
                      "starcoderbase-7b","starcoderbase-1b"]:
            for mut in ["types","vars","delete","vars_delete","types_delete","types_vars","delete_vars_types"]:
                for interval in [1,3,5]:
                    for layers in get_ranges(model_n_layer(model), interval):
                        try:
                            ds = datasets.load_dataset("nuprl-staging/type-steering-results",
                                                    f"steering-{lang}-{mut}-{layers}-{model}",split=split)
                        except Exception as e:
                            print(e)
                            continue
                        df = ds.to_pandas()
                        counts_mut = df.value_counts("mutated_generated_text")
                        counts = df.value_counts("generated_text")
                        if counts_mut.get("", 0) > 0:
                            print(f"mutations {split}: {lang}-{mut}-{layers}-{model}: {counts_mut['']}")
                        if counts.get("", 0) > 0:
                            print(f"generated {split}: {lang}-{mut}-{layers}-{model}: {counts['']}")
                        print(counts.get("", 0), counts_mut.get("", 0))
                progress_bar.update(1)
    progress_bar.close()
                
if __name__ == "__main__":
    datasets.disable_progress_bars()
    check_all_generations_in_test("test")
    check_all_generations_in_test("steer")
    check_all_generations()
    
