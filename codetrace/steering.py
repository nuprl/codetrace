from codetrace.scripts.typecheck_ds import multiproc_typecheck
from multiprocessing import cpu_count
import math
import numpy as np
from collections import Counter
import torch
from pathlib import Path
from codetrace.parsing_utils import FimObj, get_model_fim, FimChat
import datasets
import os
from transformers import PreTrainedTokenizer
from typing import Union, Tuple,List,Dict,Any,Optional,Callable
from codetrace.utils import load_dataset, save_dataset, mask_target_idx, masked_add
from nnsight import LanguageModel
from codetrace.batched_utils import batched_get_averages, batched_insert_patch_logit
from functools import partial
import itertools as it
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from codetrace.parsing_utils import prepare_fim_prompt
import glob

def balance_prompts(
    dataset : datasets.Dataset,
    dedup_prog_threshold : int,
    dedup_type_threshold : int,
    disable_tqdm:bool = False
) -> datasets.Dataset:
    """
    Balance prompts s.t. there is a balanced distribution of labels (program ids and/or types).
    """
    # if -1, set to the max value, aka do not dedup
    prog_maxn = len(dataset) if dedup_prog_threshold == -1 else dedup_prog_threshold
    type_maxn = len(dataset) if dedup_type_threshold == -1 else dedup_type_threshold
    
    program_count = {i:0 for i in dataset["_original_program"]}
    label_count = {label : 0 for label in dataset["fim_type"]}
    balanced_prompts = []
    for _,ex in tqdm(enumerate(dataset), desc="Deduping dataset",total=len(dataset), disable=disable_tqdm):
        if program_count[ex["_original_program"]] >= prog_maxn and label_count[ex["fim_type"]] >= type_maxn:
            break
        elif label_count[ex["fim_type"]] + 1 >= type_maxn or program_count[ex["_original_program"]] + 1 >= prog_maxn:
            continue
        
        balanced_prompts.append(ex)
        label_count[ex["fim_type"]] += 1
        program_count[ex["_original_program"]] += 1

    ds = datasets.Dataset.from_list(balanced_prompts)
    return ds

def subtract_avg(hidden_states:torch.Tensor) -> torch.Tensor:
    # [n_layer, n_prompt, n_tokens, n_vocab]
    """
    At prompt dim, subtract pairs of prompts; finally, compute the mean
    """
    # even and odd indices
    even_indices = torch.arange(0, hidden_states.shape[1], step=2)
    odd_indices = torch.arange(1, hidden_states.shape[1], step=2)

    # subtract odd idx from even
    hidden_states[:, even_indices] -= hidden_states[:, odd_indices]

    # compute mean
    mean_even = hidden_states[:, even_indices].mean(dim=1, keepdim=True)
    return mean_even

def prepare_prompt_pairs(data: List[Dict[str,Any]], format_fn: Callable[[str], str])->List[str]:
    preproc_fn = (lambda x: (format_fn(x["fim_program"]), format_fn(x["mutated_program"])))
    return list(it.chain.from_iterable(map(preproc_fn, data)))
    
class SteeringManager:
    """
    This class bundles methods for steering, saving and running 
    the type inference experiment given a model, a save dir and
    steering candidates (neg-pos pairs). 
    """

    def __init__(
        self,
        model:LanguageModel,
        candidates_ds: datasets.Dataset,
        cache_dir: Path,
        steer_split_path: Optional[str]=None,
        test_split_path: Optional[str]=None,
        steering_tensor_path: Optional[str]=None,
        max_num_candidates:int=-1,
        token_mask_fn:Optional[Callable]=None,
        reduction:Optional[Callable]=None,
        only_collect_layers:Optional[List[int]]=None
    ):
        self.model=model
        self.tokenizer=model.tokenizer
        self.fim_obj=get_model_fim(model.config.name_or_path)
        if max_num_candidates > -1 and max_num_candidates < len(candidates_ds):
            # select first and foremost from typechecking candidates
            candidates_ds = candidates_ds.sort("typechecks", reverse=True)
            assert candidates_ds[0]["typechecks"]
            candidates_ds = candidates_ds.select(range(max_num_candidates))
            
        self.candidates_ds = candidates_ds.map(
            lambda x: {**x, "_original_program": x["fim_program"].replace("<FILL>", x["fim_type"])},
            desc="Adding column for original unfimmed program"
        )
        self.cache_dir = cache_dir
        if not token_mask_fn:
            # default patch on fim middle last token
            token_mask_fn = partial(mask_target_idx, indices=[-1])
            reduction = "sum"
        self.patch_fn = masked_add
        self.token_mask_fn = token_mask_fn
        self.reduction = reduction
        self.only_collect_layers=only_collect_layers
        # try load cached if it exists
        self.test_split : Optional[datasets.Dataset] = self.load_data(test_split_path)
        self.steer_split : Optional[datasets.Dataset] = self.load_data(steer_split_path)
        self.steering_tensor : Optional[torch.Tensor] = self.load_tensor(steering_tensor_path)
        os.makedirs(self.cache_dir, exist_ok=True)

    def tokenize(self, prompt:str) -> str:
        return prepare_fim_prompt(self.tokenizer, self.fim_obj, prompt)

    def save_data(self, data:datasets.Dataset, path:str):
        """
        Saves data to self.cache_dir / path
        """
        subpath = Path(os.path.join(self.cache_dir, path))
        if not os.path.exists(subpath):
            save_dataset(data, subpath)

    def load_data(self, path:str, split:Optional[str]=None) -> Optional[datasets.Dataset]:
        """
        Loads data from self.cache_dir / path
        """
        try:
            return load_dataset(os.path.join(self.cache_dir,path), split=split)
        except:
            return None
    
    def save_tensor(self, tensor:torch.Tensor, path:str):
        """
        Saves tensor to self.cache_dir / path
        """
        torch.save(tensor,os.path.join(self.cache_dir, path))

    def load_tensor(self, path:str) -> Optional[torch.Tensor]:
        """
        Loads tensor from self.cache_dir / path
        """
        fullpath=os.path.join(self.cache_dir, path)
        if os.path.exists(fullpath):
            return torch.load(fullpath)
        else:
            return None

    def clear_cache(self):
        """
        Clear any cached intermediate computations once final
        computations are done
        """
        if self.steering_tensor != None:
            for file in glob.glob(os.path.join(self.cache_dir, "cached_steering_tensor.*")):
                os.remove(file)
        if self.test_split != None:
            for file in glob.glob(os.path.join(self.cache_dir, "cached_steering_*")):
                os.remove(file)

    def steer_test_splits(
        self,
        test_size:Union[float,int],
        dedup_prog_threshold:int, # 3 suggested
        dedup_type_threshold:int, # 25 suggested
        shuffle:bool=False,
        seed:Optional[int]=None,
        debug_max_cycle:Optional[int]=None
    )-> Tuple[datasets.Dataset]:
        """
        Split candidates into a steering and test split.
        Ensure that source programs in test are not in steer
        and viceversa using the program id to sort.

        Arguments:
            test_size: int for number of test programs. float for portion of test programs.
            dedup_prog_threshold: max number of duplicate source programs that should be
                present in the splits.
            dedup_type_threshold: max number of duplicate type_labels that should be
                present in the splits.
            shuffle: whether to shuffle candidates
            seed: for deterministic shuffling

        NOTE: test split is always the same for a dataset of candidates
        """
        if not self.test_split and not self.steer_split:
            if test_size < 0:
                test_size *= len(self.candidates_ds)

            steer_split,test_split = _steer_test_split(
                self.candidates_ds,
                test_size=test_size,
                shuffle=shuffle,
                seed=seed,
                separate_by_column="_original_program",
                debug_max_cycle=debug_max_cycle,
                dedup_prog_threshold=dedup_prog_threshold,
                dedup_type_threshold=dedup_type_threshold,
            )
            
            train_size = len(self.candidates_ds) - test_size
            if train_size < len(steer_split):
                steer_split = steer_split.select(range(train_size))
            if test_size < len(test_split):
                test_split = test_split.select(range(test_size))
            self.test_split = test_split
            self.steer_split = steer_split

        return self.steer_split, self.test_split

    def create_steering_tensor(self, batch_size:int) -> torch.Tensor:
        """
        Collects activations of steering split in a batched manner, subtracting
        negative and positive steering items and averaging result as it goes.
        """
        if not self.steer_split:
            raise ValueError("Please create a steer split before attempting to steer.")
        if batch_size % 2 != 0:
            raise ValueError("Please provide a batch_size divisible by pairs")
        
        if self.steering_tensor == None:
            dataloader = torch.utils.data.DataLoader(
                self.steer_split,
                batch_size,
                collate_fn=partial(prepare_prompt_pairs, format_fn=self.tokenize)
            )
            self.steering_tensor = batched_get_averages(
                self.model,
                dataloader,
                batch_size=batch_size,
                target_fn=self.token_mask_fn,
                reduction=self.reduction,
                layers=self.only_collect_layers,
                average_fn=subtract_avg,
                outfile=os.path.join(self.cache_dir, "cached_steering_tensor")
            )
        return self.steering_tensor
    
    def steer(
        self,
        split:str,
        layers_to_steer:List[int],
        batch_size:int,
        do_random_ablation:bool=False,
        steering_field:Optional[str]=None
    )-> datasets.Dataset:
        """
        Evaluate the steering tensor on data.
        If do_random_ablation is set, will run a random steering tensor.
        """
        if self.only_collect_layers and not set(layers_to_steer).issubset(set(self.only_collect_layers)):
            raise ValueError(f"Trying to steer layers {layers_to_steer} but only collected steering tensor from {self.only_collect_layers}")
        if self.steering_tensor == None:
            raise ValueError("Please create a steering tensor before attempting to steer.")
        if any([self.steering_tensor[l].sum().item() == 0 for l in layers_to_steer]):
            raise ValueError(f"Trying to steer layers {layers_to_steer} but steering tensor is empty at one of the layers.")
        if split == "steer":
            ds = self.steer_split
        elif split == "test":
            ds = self.test_split
        else:
            raise ValueError("Can only specify to steer either on the steer or test split.")
        
        if do_random_ablation:
            steering_tensor = torch.rand_like(self.steering_tensor)
            # steering_tensor.random_(math.floor(self.steering_tensor.min().item()), 
            #                       math.ceil(self.steering_tensor.max().item()))
        else:
            steering_tensor = self.steering_tensor

        steering_field = (steering_field or "mutated_program")
        dataloader = torch.utils.data.DataLoader(
            ds[steering_field],
            batch_size,
            collate_fn = (lambda x: list(map(self.tokenize, x)))
        )
        solutions = list(ds["fim_type"])
        predictions = batched_insert_patch_logit(
            self.model,
            dataloader,
            steering_tensor,
            layers_to_steer,
            target_fn=self.token_mask_fn,
            batch_size=batch_size,
            outfile=os.path.join(self.cache_dir, f"cached_steering_{split}"),
            solutions=solutions,
            patch_fn=self.patch_fn
        )
        ds = ds.add_column("steered_predictions",predictions)
        return ds

def _steer_test_split(
    ds: datasets.Dataset,
    test_size: Union[int, float],
    shuffle:bool,
    seed:Optional[int],
    separate_by_column: str,
    debug_max_cycle: Optional[int] = None,
    dedup_prog_threshold: int = -1,
    dedup_type_threshold: int = -1,
)-> Tuple[datasets.Dataset, datasets.Dataset]:
    if shuffle:
        ds = ds.shuffle(seed=seed)
    unique_labels = np.unique(ds[separate_by_column])
    print(len(unique_labels))
    actual_test_size = test_size if test_size > 0 else int(test_size * len(ds))
    test_ds, i = [],0
    INCREASE_AMT = 10
    while len(test_ds) < actual_test_size:
        print(f"Attempting split #{i}")
        if debug_max_cycle and i > debug_max_cycle:
            break
        train_labels, test_labels = train_test_split(
                                unique_labels, 
                                test_size=actual_test_size+(i*INCREASE_AMT),
                                shuffle=shuffle)
        train_ds = ds.filter(lambda x: x[separate_by_column] in train_labels, num_proc=10)
        test_ds = ds.filter(lambda x: x[separate_by_column] in test_labels and x["typechecks"], num_proc=10)
        
        if dedup_prog_threshold > -1 or dedup_type_threshold > -1:
            train_ds = balance_prompts(train_ds, dedup_prog_threshold, dedup_type_threshold)
            test_ds = balance_prompts(test_ds, dedup_prog_threshold, dedup_type_threshold)
        
        print(f"Split size:", len(train_ds), len(test_ds))
        i += 1
    return train_ds, test_ds