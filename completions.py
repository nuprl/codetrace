from vllm import LLM, SamplingParams
import datasets
import json
import pandas as pd

starcoderbase_1b = "/home/arjun/models/starcoderbase-1b/"
llm = LLM(starcoderbase_1b)

dataset = datasets.load_dataset("franlucc/ts_bench_starcoder1b_funcfim_incorrect_uniq", split="train")

sampling_params = SamplingParams(temperature=0, max_tokens=1)

out = llm.generate([ex["prompt"] for ex in dataset], sampling_params)

res = []
new_ds = []
for i,ex in enumerate(out):
    prediction = ex.outputs[0].text
    res.append({"idx" : i,
                "solution" : dataset[i]["fim_sol"],
                "generated" : dataset[i]["generated_text"],
                "prediction" : prediction})
    if dataset[i]["fim_sol"] != prediction:
        d = dataset[i]
        d["generated_text"] = prediction
        new_ds.append(d)
with open("predictions.json", "w") as f:
    json.dump(res, f)

hf_new_ds = datasets.Dataset.from_pandas(pd.DataFrame(new_ds))
hf_new_ds.push_to_hub("franlucc/ts_bench_starcoder1b_funcfim_incorrect_uniq_corrected")