import multiprocessing
from tqdm import tqdm
import datasets
from typing import List, Iterable,Any,Callable
"""
Fast parallel utils for processing data in batches.
Faster than huggingface multiproc map/filter.
"""
def batched_apply(batches : List[List[Any]], num_proc : int, func : Callable, **func_kwargs) -> List[Any]:
    """
    Apply a function to batches of data in parallel.
    A batch is a list of data.
    """
    pool = multiprocessing.Pool(num_proc)
    disable_tqdm = func_kwargs.pop("disable_tqdm",False)
    desc = func_kwargs.pop("desc","Applying")

    async_out_batches = []
    for i, batch in enumerate(batches):
        async_out = pool.apply_async(func, args=(batch,), kwds=func_kwargs)
        async_out_batches.append(async_out)
    
    results = []
    for i in tqdm(range(len(async_out_batches)),desc=desc,disable=disable_tqdm):
        results += async_out_batches[i].get()
            
    pool.close()
    pool.join()
    return results

def _collect_index(itr: Iterable, si:int, ei:int) -> Iterable:
    if isinstance(itr, datasets.Dataset):
        return list(itr.select(range(si, ei))) # faster
    else:
        return itr[si:ei]
    
def make_batches(iterable:Iterable, num_proc : int, disable_tqdm: bool = True) -> List[Iterable]:
    """
    Makes batches of data for parallel processing.
    Is parallelized for speed on large datasets.
    Does not preserve order.
    """
    batch_size = (len(iterable) // num_proc) or 1
    pool = multiprocessing.Pool(num_proc)
    async_out_batches = []
    
    i=0
    while i < len(iterable):
        end_index = min(i + batch_size, len(iterable))
        async_out = pool.apply_async(_collect_index, args=(iterable, i, end_index))
        async_out_batches.append(async_out)
        i = end_index
    
    batches = []
    for j in tqdm(range(len(async_out_batches)),desc="Batching",disable=disable_tqdm):
        batches.append(async_out_batches[j].get())
        
    pool.close()
    pool.join()
    return batches