import numpy as np
from collections import Counter
import torch
from pathlib import Path
from codetrace.parsing_utils import FimObj, get_model_fim
import datasets
import os
from typing import Union, Tuple,List,Dict,Any,Optional,Callable
from codetrace.utils import load, save, mask_target_tokens, keep_columns
from nnsight import LanguageModel
from codetrace.type_inf_exp.batched_utils import batched_get_averages, batched_insert_patch_logit
from functools import partial
from codetrace.fast_utils import batched_apply, make_batches
import itertools as it
from tqdm import tqdm
from sklearn.model_selection import train_test_split

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
    if dedup_prog_threshold == -1:
        dedup_prog_threshold = len(dataset)
    if dedup_type_threshold == -1:
        dedup_type_threshold = len(dataset)
        
    # get count of labels
    labels = dataset["fim_type"]
    
    hexsha_count = {h:0 for h in dataset["hexsha"]}
    label_count = {label : 0 for label in labels}
    balanced_prompts = []
    for _,ex in tqdm(enumerate(dataset), desc="Deduping dataset",total=len(dataset), disable=disable_tqdm):
        if label_count[ex["fim_type"]] >= dedup_type_threshold and hexsha_count[ex["hexsha"]] >= dedup_prog_threshold: 
            # if label and hexsha are already at threshold, break
            break
        elif label_count[ex["fim_type"]] >= dedup_type_threshold or hexsha_count[ex["hexsha"]] >= dedup_prog_threshold:
            # if hexsha is at threshold, continue
            continue
        
        balanced_prompts.append(ex)
        label_count[ex["fim_type"]] += 1
        hexsha_count[ex["hexsha"]] += 1

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
    mean_even = hidden_states[:, even_indices].mean(dim=1)
    return mean_even

def prepare_prompts(
    data: List[Dict[str,Any]],
    fim_obj:FimObj
)->List[str]:
    return list(it.chain.from_iterable(
                    map(lambda x:(fim_obj.placeholder_to_fim(x["fim_program"]),
                                fim_obj.placeholder_to_fim(x["mutated_program"])), data),
                    )
                )
    
def multiproc_prepare_prompts(fim_programs:List[Dict[str,Any]], fim_obj:FimObj, nproc:int=2)->List[str]:
    batches = make_batches(fim_programs, nproc, disable_tqdm=True)
    return batched_apply(batches, nproc, prepare_prompts, disable_tqdm=True, fim_obj=fim_obj)

class SteeringManager:
    """
    This class bundles methods for steering, saving and running 
    the type inference experiment given a model, a save dir and
    steering candidates (neg-pos pairs). 
    """

    def __init__(
        self,
        model:LanguageModel,
        candidates_ds: str,
        cache_dir: Path,
        steer_split_path: Optional[str]=None,
        test_split_path: Optional[str]=None,
        steering_tensor_path: Optional[str]=None,
        max_num_candidates:int=-1,
        token_mask_fn:Optional[Callable]=None
    ):
        self.model=model
        self.tokenizer=model.tokenizer
        self.fim_obj=get_model_fim(model.config.name_or_path)
        self.candidates_ds = load(candidates_ds)
        if max_num_candidates > -1:
            self.candidates_ds = self.candidates_ds.select(range(max_num_candidates))
        self.cache_dir = cache_dir
        if not token_mask_fn:
            # default patch on fim middle
            token_mask_fn = partial(mask_target_tokens, tokens=[self.fim_obj.middle], tokenizer=self.tokenizer),
        self.token_mask_fn = token_mask_fn
        # try load cached if it exists
        self.test_split : Optional[datasets.Dataset] = self.load_data(test_split_path)
        self.steer_split : Optional[datasets.Dataset] = self.load_data(steer_split_path)
        self.steering_tensor : Optional[torch.Tensor] = self.load_tensor(steering_tensor_path)

        os.makedirs(self.cache_dir, exist_ok=True)

    def save_data(self, data:datasets.Dataset, path:str):
        """
        Saves data to self.cache_dir / path
        """
        subpath = Path(os.path.join(self.cache_dir, path))
        if not os.path.exists(subpath):
            save(data, subpath)

    def load_data(self, path:str, split:Optional[str]=None) -> Optional[datasets.Dataset]:
        """
        Loads data from self.cache_dir / path
        """
        try:
            return load(os.path.join(self.cache_dir,path), split=split)
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

    def steer_test_splits(
        self,
        test_size:Union[float,int],
        dedup_prog_threshold:int, # 3 suggested
        dedup_type_threshold:int, # 25 suggested
        shuffle:bool=True,
        seed:Optional[int]=None
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
        """
        if not self.test_split and not self.steer_split:
            steer_split,test_split = _steer_test_split(
                self.candidates_ds,
                test_size=test_size,
                shuffle=shuffle,
                seed=seed,
                separate_by_column="hexsha"
            )
            if dedup_prog_threshold > -1 or dedup_type_threshold > -1:
                steer_split = balance_prompts(steer_split, dedup_prog_threshold, dedup_type_threshold)
                test_split = balance_prompts(steer_split, dedup_prog_threshold, dedup_type_threshold)
            
            if test_size < 0:
                test_size *= len(self.candidates_ds)
            train_size = len(self.candidates_ds) - test_size
            test_size = min(len(test_split), test_size)
            train_size = min(len(steer_split), train_size)

            self.steer_split = steer_split.select(range(train_size))
            self.test_split = test_split.select(range(test_size))

        return self.steer_split, self.test_split

    def create_steering_tensor(
        self,
        batch_size:int
    ) -> torch.Tensor:
        """
        Collects activations of steering split in a batched manner, subtracting
        negative and positive steering items and averaging result as it goes.
        """
        if not self.steer_split:
            raise ValueError("Please create a steer split before attempting to steer.")
        if batch_size % 2 != 0:
            raise ValueError("Please provide a batch_size divisible by pairs")
        
        if not self.steering_tensor:
            dataloader = torch.utils.data.DataLoader(
                self.steer_split,
                batch_size,
                collate_fn=partial(prepare_prompts, fim_obj=self.fim_obj)
            )
            self.steering_tensor = batched_get_averages(
                self.model,
                dataloader,
                batch_size=batch_size,
                target_fn=partial(mask_target_tokens, tokens=[self.fim_obj.middle], tokenizer=self.tokenizer),
                average_fn=subtract_avg,
                outfile=os.path.join(self.cache_dir, "cached_steering_tensor")
            )
        return self.steering_tensor
    
    def steer(
        self,
        split:str,
        layers_to_steer:List[int],
        batch_size:int,
    )-> datasets.Dataset:
        """
        Evaluate the steering tensor on
        """
        if not self.steering_tensor:
            raise ValueError("Please create a steering tensor before attempting to steer.")
        if split == "steer":
            ds = self.steer_split
        elif split == "test":
            ds = self.test_split
        else:
            raise ValueError("Can only specify to steer either on the steer or test split.")
        
        dataloader = torch.utils.data.DataLoader(
            ds["mutated_program"],
            batch_size,
            collate_fn=(lambda x: list(map(self.fim_obj.placeholder_to_fim, x)))
        )
        solutions = list(ds["fim_type"])
        predictions = batched_insert_patch_logit(
            self.model,
            dataloader,
            self.steering_tensor,
            layers_to_steer,
            target_fn=self.token_mask_fn,
            batch_size=batch_size,
            outfile=os.path.join(self.cache_dir, f"cached_steering_{split}"),
            solutions=solutions
        )
        ds["steered_predictions"] = predictions
        return ds

def _steer_test_split(
    ds: datasets.Dataset,
    test_size: Union[int, float],
    shuffle:bool,
    seed:Optional[int],
    separate_by_column: str,
)-> Tuple[datasets.Dataset, datasets.Dataset]:
    """
    referenced from huggingface stratified_shuffle_split_generate_indices
    """
    counter = Counter(ds[separate_by_column])
    ds = ds.map(lambda x: {**x, "_label": x[separate_by_column] if counter[x[separate_by_column]] > 1 else None},
                                        num_proc=10)
    ds = ds.class_encode_column("_label", include_nulls=True)
    if shuffle:
        ds = ds.shuffle(seed=seed)

    unique_labels = np.unique(ds[separate_by_column])
    train_labels, test_labels = train_test_split(unique_labels, test_size=test_size)

    train_ds = ds.filter(lambda x: x[separate_by_column] in train_labels, num_proc=10)
    test_ds = ds.filter(lambda x: x[separate_by_column] in test_labels, num_proc=10)
    return train_ds, test_ds


"""
PYTESTS
"""

def test_subtract_avg():
    x = torch.Tensor(
        [
            # layer0
            [
                # prompt 0
                [[1,2],[3,4],[5,6]],
                # prompt 1
                [[1,2],[3,4],[5,6]],
            ],

            # layer1
            [
                # prompt 0
                [[1,2],[3,4],[5,6]],
                # prompt 1
                [[-1,-2],[-3,-4],[-5,-6]],
            ],
            # layer2
            [
                # prompt 0
                [[1,2],[3,4],[5,6]],
                # prompt 1
                [[1,-2],[3,-4],[-5,6]],
            ],
        ]
    )
    output = subtract_avg(x)
    expected = torch.Tensor(
        [
            # layer0
            [
                [0,0],[0,0],[0,0],
            ],

            # layer1
            [
                # prompt 0
                [2,4],[6,8],[10,12],
            ],
            # layer2
            [
                [0,4],[0,8],[10,0],
            ],
        ]).to(dtype=torch.float)
    assert torch.equal(output, expected), f"{output} != {expected}"

def test_prepare_prompts():
    prompts = [
        {"fim_program":"my name is <FILL> !","mutated_program":"my name is NOT <FILL> !"},
        {"fim_program":"my job is <FILL> !","mutated_program":"my job is NOT <FILL> !"},
        {"fim_program":"my house is <FILL> !","mutated_program":"my house is NOT <FILL> !"},
        {"fim_program":"my car is <FILL> !","mutated_program":"my car is NOT <FILL> !"},
    ]
    fim_obj = get_model_fim("starcoder")
    output = prepare_prompts(prompts, fim_obj)
    expected = []
    for i in prompts:
        expected.append(fim_obj.placeholder_to_fim(i["fim_program"]))
        expected.append(fim_obj.placeholder_to_fim(i["mutated_program"]))
    assert output == expected, f"{output}!={expected}"

def test_stratify():
    data = [{"hexsha":0},
            {"hexsha":1},
            {"hexsha":2},
            {"hexsha":3},
            {"hexsha":4},
            {"hexsha":5},
            {"hexsha":5},
            {"hexsha":5},
            {"hexsha":6},
            {"hexsha":6},
            {"hexsha":7}]
    ds = datasets.Dataset.from_list(data)
    steer_split,test_split = _steer_test_split(
        ds,
        4,
        True,
        None,
        "hexsha"
    )
    steer_split = [x["hexsha"] for x in steer_split]
    test_split = [x["hexsha"] for x in test_split]
    print(steer_split, test_split)
    assert set(steer_split).intersection(set(test_split)) == set(), f"{steer_split} - {test_split}"