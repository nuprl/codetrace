
from collections import defaultdict
import asyncio
import datasets
import argparse
import uuid
import random
from multiprocessing import cpu_count
from vllm import AsyncLLMEngine, SamplingParams, AsyncEngineArgs
from vllm.outputs import RequestOutput
import csv
from tqdm import tqdm
from codetrace.type_inf_exp.py_mutator import incremental_mutate
import os
from codetrace.type_inf_exp import py_mutator 
from codetrace.parsing_utils import get_model_fim, placeholder_to_std_fmt, FimObj, std_to_placeholder_fmt
from codetrace.utils import load_dataset, save_dataset, num_available_devices
from typing import List, Tuple, Generator,AsyncGenerator,Union
from dataclasses import dataclass
import threading
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

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
    model_fim:FimObj,
    max_n:int=10 # abandon robust prompts
)->Generator:
    mut_prompts = incremental_mutate(prompt, fim_type, mutations)
    i = 0
    for mp in mut_prompts:
        if mp != None:
            yield placeholder_to_std_fmt(mp, model_fim)
            i += 1
            if max_n>0 and i>max_n:
                break

async def launch_generation(
    request_generator: ResultGeneratorWrapper
) -> AsyncGenerator:
    async for request_output in request_generator.generator:
        prompt = request_output.prompt
        assert prompt is not None
        for out_idx,output in enumerate(request_output.outputs):
            if output.text != "" and output.text != None:
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
    stop_event: threading.Event,
    llm: AsyncLLMEngine,
    mutations: List[callable],
    model_fim:FimObj
):
    sampling_params = SamplingParams(temperature=0, max_tokens=1)
    for idx,item in tqdm(enumerate(ds), desc="Producing requests", total=len(ds)):
        if stop_event.is_set():
            break
        mut_prompts = mutation_generator(item["fim_program"], item["fim_type"], mutations, model_fim)
        for mut in mut_prompts:
            request_id = uuid.uuid4().int
            request = llm.generate(mut, sampling_params, request_id=request_id)
            async for gen in launch_generation(ResultGeneratorWrapper(idx, request, request_id)):
                await queue.put(gen)
                
    # no more examples to mine, stop with what we have
    await asyncio.sleep(60) # let consumer finish
    stop_event.set()

def temp_save(log_path:str, data:dict):
    path = f"{log_path}.csv"
    file_exists = os.path.isfile(path)
    with open(path,mode="a", newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=data.keys())

        # Write the header only if the file is new (doesn't exist yet)
        if not file_exists:
            writer.writeheader()

        # Append the dictionary as a new row
        writer.writerow(data)


async def request_consumer(
    ds: datasets.Dataset,
    queue: asyncio.Queue,
    result_queue: asyncio.Queue,
    stop_event: threading.Event,
    llm: AsyncLLMEngine,
    num_examples:int,
    model_fim:FimObj,
    log_path:str,
    candidate_fn:callable
)-> datasets.Dataset:
    done_idx = set()
    idx_to_request_ids = defaultdict(list)
    consume_bar = tqdm(desc="Consuming requests")
    progress_bar = tqdm(range(num_examples), desc="Items")
    while result_queue.qsize() < num_examples and not stop_event.is_set():
        result= await queue.get()

        dataset_idx = result.dataset_idx
        idx_to_request_ids[dataset_idx].append(result.request_id)

        if dataset_idx in done_idx:
            # since requests come asynchronously, keep aborting procesed items
            abort_request_ids(llm,idx_to_request_ids[dataset_idx])
            continue
        
        prompt = result.prompt
        generated_text = result.generated
        data = {
                **ds[dataset_idx],
                "item_idx":result._output_idx,
                "mutated_program": std_to_placeholder_fmt(prompt, model_fim), 
                "mutated_generated_text": generated_text
            }
        # temporary save
        temp_save(log_path + "_log",data)

        if candidate_fn(generated_text.strip(),ds[dataset_idx]["fim_type"]):
            await result_queue.put(data)
            abort_request_ids(llm,idx_to_request_ids[dataset_idx])
            done_idx.add(dataset_idx)
            progress_bar.update(1)
            # temporary save
            temp_save(log_path,data)
    
        # give producer generation some more time randomly, in practice this speeds
        # things up because GPU utilization is driven up
        await asyncio.sleep(random.uniform(0.1,2.5))
        # done
        consume_bar.update(1)
        queue.task_done()

    stop_event.set()
    consume_bar.close()
    progress_bar.close()
    return result_queue

async def main(
    completions_ds:str,
    model:str,
    tokenizer:str,
    dtype:str,
    new_ds_name:str,
    num_examples:int,
    mutations:List[str],
    split:str = None,
    max_size:int = -1,
    correct_bool:bool=True,
    log_requests:bool=False
):
    ds = load_dataset(completions_ds, split=split).shuffle()
    mutations = [getattr(py_mutator, m) for m in mutations]

    if correct_bool:
        candidate_fn = (lambda x,y: x.strip()!=y.strip())
    else:
        candidate_fn = (lambda x,y: x.strip()==y.strip())

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
        dtype=dtype,
        tokenizer=model if not tokenizer else tokenizer
    )
    llm = AsyncLLMEngine.from_engine_args(engine_args)
    llm.log_requests = log_requests    
    model_fim = get_model_fim(model)

    # init log files
    for file_path in [f"{new_ds_name}.csv", f"{new_ds_name}_log.csv"]:
        if os.path.exists(file_path):
            os.remove(file_path)
    """
    For each prompt, generate N mutation combinations until one breaks the model,
    then go onto the next prompt. Do this asynchronously so while model is
    generating a request, we can process the previous one. This maximises gpu utilization
    while retrieving results.
    """
    queue = asyncio.Queue()
    stop_event = threading.Event()
    result_queue = asyncio.Queue()

    producer_task = asyncio.create_task(request_producer(ds,queue,stop_event,llm,mutations,model_fim))
    consumer_task = asyncio.create_task(request_consumer(ds,queue,result_queue,stop_event,llm,
                                                         num_examples,model_fim, new_ds_name,candidate_fn))
    await asyncio.gather(producer_task, consumer_task)

    new_ds = []
    while not result_queue.empty():
        result = await result_queue.get()
        new_ds.append(result)
        result_queue.task_done()

    new_ds = datasets.Dataset.from_list(new_ds)
    print(new_ds)
    save_dataset(new_ds, new_ds_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--completions-ds", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--dtype", type=str, required=True, choices=["bfloat16", "float32"])
    parser.add_argument("--new-ds-name", type=str, required=True)
    parser.add_argument("--num-examples","-n", type=int, required=True)
    parser.add_argument("--mutations", type=str, required=True, nargs="+", choices=["mutation_rename_type",
                                                                                    "mutation_rename_vars",
                                                                                    "mutation_delete_annotation"])
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--max-size", type=int, default=-1)
    parser.add_argument("--do-incorrect", action="store_true")
    parser.add_argument("--log-requests", action="store_true")
    args = parser.parse_args().__dict__
    args["correct_bool"] = not args.pop("do_incorrect", True)
    datasets.disable_caching()
    print("Correct:", args["correct_bool"])
    print("Caching enabled?:", datasets.is_caching_enabled())
    print("Gpu:", os.environ["CUDA_VISIBLE_DEVICES"])
    asyncio.run(main(**args))
