import multiprocessing
from tqdm import tqdm
import datasets
"""
Fast parallel utils for processing data in batches.
Faster than huggingface multiproc map/filter.
"""
    
def batched_do_func(batches, num_proc, func, **func_kwargs):
    """
    Apply a function to batches of data in parallel.
    """
    pool = multiprocessing.Pool(num_proc)
    
    async_out_batches = []
    for i, batch in tqdm(enumerate(batches), desc="Processing batches", total=len(batches)):
        async_out = pool.apply_async(func, args=(batch,), kwds=func_kwargs)
        async_out_batches.append(async_out)
    
    results = []
    for i in tqdm(range(len(async_out_batches)), desc="Getting results", total=len(async_out_batches)):
        results += async_out_batches[i].get()
            
    pool.close()
    pool.join()
    return results

def _collect_index(itr, si, ei):
    if isinstance(itr, datasets.Dataset):
        return [itr[i] for i in range(si, ei)] # faster
    else:
        return itr[si:ei]
      
def get_batches_fast(iterable, len_iter, num_proc):
    """
    Not for IterableDataset. Makes batches of data for parallel processing.
    Is parallelized for speed on large datasets.
    """
    batch_size = len_iter // num_proc
    if batch_size < 1:
        batch_size = 1
    pool = multiprocessing.Pool(num_proc)
    async_out_batches = []
    
    i=0
    progress_bar = tqdm(total=num_proc, desc="Making batches")
    while i < len_iter:
        end_index = min(i + batch_size, len_iter)
        async_out = pool.apply_async(_collect_index, args=(iterable, i, end_index))
        async_out_batches.append(async_out)
        i = end_index
        progress_bar.update(1)
    progress_bar.close()
    
    batches = []
    for j in tqdm(range(len(async_out_batches)), desc="Getting batches", total=len(async_out_batches)):
        batches.append(async_out_batches[j].get())
        
    pool.close()
    pool.join()
    return batches