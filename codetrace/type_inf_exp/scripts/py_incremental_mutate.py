import asyncio
from fastapi.responses import StreamingResponse
import datasets
import argparse
import uuid
from multiprocessing import cpu_count
from vllm import LLM, AsyncLLMEngine, SamplingParams, AsyncEngineArgs
from tqdm import tqdm
from codetrace.fast_utils import get_batches_fast, batched_do_func
from codetrace.type_inf_exp.py_mutator import incremental_mutate
import os
from codetrace.type_inf_exp import py_mutator 
from codetrace.parsing_utils import get_model_fim, placeholder_to_std_fmt, FimObj, std_to_placeholder_fmt
from codetrace.utils import load, save, num_available_devices
from typing import List, Tuple, Generator,AsyncGenerator
import itertools as it
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

async def chain_generators(generators:List[AsyncGenerator]) ->AsyncGenerator:
    for gen in generators:
        async for item in gen:
            yield item

async def iterate_async_requests(
    requests_generators: List[AsyncGenerator]
) -> AsyncGenerator:
    async for request_output in chain_generators(requests_generators):
        prompt = request_output.prompt
        assert prompt is not None
        text_outputs = [
            (prompt, output.text) for output in request_output.outputs
        ]
        for item in text_outputs:
            yield item

# https://github.com/vllm-project/vllm/blob/b9c64c0ca79ccdea608f337fbb7e4b0c75fe3aac/vllm/engine/async_llm_engine.py#L227
def abort_all_requests(llm:AsyncLLMEngine):
    while not llm._request_tracker._new_requests.empty():
        stream, _ = llm._request_tracker._new_requests.get_nowait()
        request_id = stream.request_id
        llm.abort_request(request_id)

def mutation_generator(
    prompt:str,
    fim_type:str,
    mutations:List[str],
    model_fim:FimObj
)->Generator:
    mut_prompts = incremental_mutate(prompt, fim_type, mutations)
    for mp in mut_prompts:
        if mp != None:
            yield placeholder_to_std_fmt(mp, model_fim)

async def main(
    completions_ds:str,
    model:str,
    dtype:str,
    new_ds_name:str,
    num_examples:int,
    mutations:List[str],
    split:str = None,
    max_size:int = -1,
    correct_bool:bool=True,
    log_requests:bool=False,
    save_every:int=100,
):
    ds = load(completions_ds, split=split)
    if max_size > -1:
        ds = ds.shuffle().select(range(max_size))
    mutations = [getattr(py_mutator, m) for m in mutations]

    # filter dataset candidates
    ds = ds.filter(lambda x: x["correct"] == correct_bool, num_proc=cpu_count(), desc="Filtering candidates")
    print(ds)

    tps = num_available_devices()
    print(f"Serving VLLM across {tps} GPUs.")
    engine_args = AsyncEngineArgs(
        model=model, 
        tensor_parallel_size=tps,
        dtype=dtype
    )
    llm = AsyncLLMEngine.from_engine_args(engine_args)
    llm.log_requests = log_requests    
    sampling_params = SamplingParams(temperature=0, max_tokens=1)

    """
    For each prompt, generate N mutation combinations until one breaks the model,
    then go onto prompt
    """
    model_fim = get_model_fim(model)
    new_ds = []
    progress_bar = tqdm(desc="Collected mutated programs", total=num_examples)
    for i, item in tqdm(enumerate(ds), desc="Iterating over candidates"):
        mut_prompts = mutation_generator(item["fim_program"], item["fim_type"], mutations, model_fim)
        results_generators = []
        for mut in mut_prompts:
            request_id = uuid.uuid4().int
            request = llm.generate(mut, sampling_params, request_id=request_id)
            results_generators.append(request)

        async for prompt,generated_text in iterate_async_requests(results_generators):
            if generated_text.strip() != item["fim_type"]:
                new_ds.append({**item, 
                    "mutated_program": std_to_placeholder_fmt(prompt, model_fim), 
                    "mutated_generated_text": generated_text})
                abort_all_requests(llm)
                progress_bar.update(1)
                break
        
        # save
        if len(new_ds)>0 and i % save_every == 0:
            save(datasets.Dataset.from_list(new_ds), new_ds_name + "_" + model.split("/")[-1])

        # end generation
        if len(new_ds) >= num_examples:
            abort_all_requests(llm)
            progress_bar.close()
            break

    new_ds = datasets.Dataset.from_list(new_ds)
    print(new_ds["fim_type"], new_ds["mutated_generated_text"])
    save(new_ds, new_ds_name + "_" + model.split("/")[-1])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--completions-ds", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dtype", type=str, required=True, choices=["bfloat16", "float32"])
    parser.add_argument("--new-ds-name", type=str, required=True)
    parser.add_argument("--num-examples","-n", type=int, required=True)
    parser.add_argument("--mutations", type=str, required=True, nargs="+", choices=["mutation_rename_type",
                                                                                    "mutation_rename_vars",
                                                                                    "mutation_delete_annotation"])
    parser.add_argument("--no-caching", action="store_true", default=False)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--max-size", type=int, default=-1)
    parser.add_argument("--correct-bool", type=bool, default=True)
    parser.add_argument("--log-requests", type=bool, default=False)
    parser.add_argument("--save-every", type=int, default=100)
    args = parser.parse_args().__dict__
    if args.pop("no_caching"):
        datasets.disable_caching()
        print("Caching enabled?:", datasets.is_caching_enabled())
    print("Gpu:", os.environ["CUDA_VISIBLE_DEVICES"])
    asyncio.run(main(**args))
