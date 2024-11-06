from concurrent.futures import ProcessPoolExecutor
from functools import partial
from collections import defaultdict
import asyncio
from fastapi.responses import StreamingResponse
import datasets
import argparse
import uuid
from aiomultiprocess import Pool
import random
from multiprocessing import cpu_count
from vllm import LLM, AsyncLLMEngine, SamplingParams, AsyncEngineArgs
from vllm.outputs import RequestOutput
from tqdm import tqdm
from codetrace.fast_utils import get_batches_fast, batched_do_func
from codetrace.type_inf_exp.py_mutator import incremental_mutate
import os
from codetrace.type_inf_exp import py_mutator 
from codetrace.parsing_utils import get_model_fim, placeholder_to_std_fmt, FimObj, std_to_placeholder_fmt
from codetrace.utils import load, save, num_available_devices, get_vllm_config
from typing import List, Tuple, Generator,AsyncGenerator,Union
import itertools as it
from dataclasses import dataclass
import multiprocessing
import aiohttp
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

class ShutdownQueue(asyncio.Queue):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shutdown_flag = False

    def shutdown(self):
        self.shutdown_flag = True

    def get(self):
        if self.shutdown_flag:
            raise asyncio.Empty
        return super().get()

@dataclass
class ResultGeneratorWrapper:
    dataset_idx:int
    generator:AsyncGenerator[RequestOutput,None]
    request_id:int 

@dataclass
class ResultWrapper:
    dataset_idx:int
    prompt:str
    generated:str
    request:RequestOutput
    request_id:int 
    _output_idx:int

# vllm/engine/async_llm_engine.py#L227
def abort_all_requests(llm:AsyncLLMEngine):
    while not llm._request_tracker._new_requests.empty():
        stream, _ = llm._request_tracker._new_requests.get_nowait()
        request_id = stream.request_id
        llm._request_tracker.abort_request(request_id)

def abort_request_ids(
    llm:AsyncLLMEngine,
    request_ids:List[int]
):
    for request_id in request_ids:
        llm._request_tracker.abort_request(request_id)
        

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

async def launch_generation(
    request_generator: ResultGeneratorWrapper
) -> AsyncGenerator:
    async for request_output in request_generator.generator:
        prompt = request_output.prompt
        assert prompt is not None
        for out_idx,output in enumerate(request_output.outputs):
            yield ResultWrapper(
                request_generator.dataset_idx,
                prompt,
                output.text, 
                request_output, 
                request_output.request_id, 
                out_idx
            )

async def request_producer(
    ds: datasets.Dataset,
    queue: asyncio.Queue,
    llm: AsyncLLMEngine,
    mutations: List[callable],
    model_fim:FimObj
):
    sampling_params = SamplingParams(temperature=0, max_tokens=1)
    for idx,item in tqdm(enumerate(ds), desc="Producing requests", total=len(ds)):
        mut_prompts = mutation_generator(item["fim_program"], item["fim_type"], mutations, model_fim)
        for mut in mut_prompts:
            request_id = uuid.uuid4().int
            request = llm.generate(mut, sampling_params, request_id=request_id)
            async for gen in launch_generation(ResultGeneratorWrapper(idx, request, request_id)):
                await queue.put(gen)

async def request_consumer(
    ds: datasets.Dataset,
    queue: asyncio.Queue,
    llm: AsyncLLMEngine,
    num_examples:int,
    model_fim:FimObj
)-> datasets.Dataset:
    new_ds = []
    done_idx = set()
    idx_to_request_ids = defaultdict(list)
    consume_bar = tqdm(desc="Consuming requests")
    progress_bar = tqdm(range(num_examples), desc="Items")
    while len(new_ds) < num_examples:
        result= await queue.get()

        if len(new_ds) >= num_examples:
            break

        dataset_idx = result.dataset_idx
        idx_to_request_ids[dataset_idx].append(result.request_id)

        if dataset_idx in done_idx:
            abort_request_ids(llm,idx_to_request_ids[dataset_idx])
            continue

        prompt = result.prompt
        generated_text = result.generated
        if generated_text.strip() != ds[dataset_idx]["fim_type"]:
            new_ds.append({
                **ds[dataset_idx], 
                "mutated_program": std_to_placeholder_fmt(prompt, model_fim), 
                "mutated_generated_text": generated_text
            })
            abort_request_ids(llm,idx_to_request_ids[dataset_idx])
            done_idx.add(dataset_idx)
            progress_bar.update(1)
        
        # give producer generation some more time randomly, in practice this speeds
        # things up because GPU utilization is driven up
        await asyncio.sleep(random.uniform(0.1,2))
        # done
        consume_bar.update(1)
        queue.task_done()

    queue.shutdown()
    consume_bar.close()
    progress_bar.close()
    return datasets.Dataset.from_list(new_ds)

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
    log_requests:bool=False
):
    ds = load(completions_ds, split=split).shuffle()
    mutations = [getattr(py_mutator, m) for m in mutations]

    # filter dataset candidates
    ds = ds.filter(lambda x: x["correct"] == correct_bool, num_proc=cpu_count(), desc="Filtering candidates")
    if max_size > -1:
        ds = ds.select(range(max_size))
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
    model_fim = get_model_fim(model)

    """
    For each prompt, generate N mutation combinations until one breaks the model,
    then go onto the next prompt. Do this asynchronously so while model is
    generating a request, we can process the previous one. This maximises gpu utilization
    while retrieving results.
    """
    queue = ShutdownQueue()
    new_ds = []

    producer_task = asyncio.create_task(request_producer(ds,queue,llm,mutations,model_fim))
    consumer_task = asyncio.create_task(request_consumer(ds,queue,llm,num_examples,model_fim))
    await asyncio.gather(producer_task, consumer_task)
    await queue.join()

    new_ds = consumer_task.result()
    consumer_task.cancel()
    print(new_ds)
    print(new_ds["mutated_generated_text"])
    save(new_ds, new_ds_name)
    abort_all_requests(llm)



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
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--max-size", type=int, default=-1)
    parser.add_argument("--correct-bool", type=bool, default=True)
    parser.add_argument("--log-requests", action="store_true")
    args = parser.parse_args().__dict__
    datasets.disable_caching()
    print("Caching enabled?:", datasets.is_caching_enabled())
    print("Gpu:", os.environ["CUDA_VISIBLE_DEVICES"])
    asyncio.run(main(**args))
