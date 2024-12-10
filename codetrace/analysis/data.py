import os
import subprocess
import itertools as it
from multiprocessing import Pool, cpu_count
from codetrace.scripts.typecheck_ds import multiproc_typecheck
import re
from abc import ABC,abstractmethod
from typing import Optional,List,Tuple,Dict,Set
import datasets
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from codetrace.utils import print_color
import pickle
from functools import wraps
from codetrace.analysis.utils import (
    ALL_MUTATIONS, 
    ALL_MODELS,
    MUTATIONS_RENAMED,
    build_success_df,
    model_n_layer,
    get_model_name,
    remove_warnings,
    remove_filename
)

HELP="""
Contains *efficient* loading and data processing script for dealing with results
of steering experiments. Can load from hub, from a local repo cloned
from hub, or from raw experiment results directory. Loads according
to passed ResultsKeys; if a key is not passed, loads all possible
values of key (eg. if 'mutations' is not passed, will load all)

See `test_result_loader` for how to load data.
""".strip()

"""
Ccahing helpers
"""

def cache_to_dir(cache_dir):
    """
    A decorator to cache function results to a specified directory.
    
    Parameters:
        cache_dir (str): The directory where cache files are stored.
    """
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create a unique filename based on function name and arguments
            cache_key = f"{func.__name__}_{hash((args, frozenset(kwargs.items())))}.pkl"
            cache_path = os.path.join(cache_dir, cache_key)
            
            # Check if the result is already cached
            if os.path.exists(cache_path):
                print(f"Loading cached result for {func.__name__} from {cache_path}")
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            
            # Compute the result and save it to the cache
            print(f"Computing result for {func.__name__} and caching to {cache_path}")
            result = func(*args, **kwargs)
            with open(cache_path, 'wb') as f:
                pickle.dump(result, f)
            return result
        
        return wrapper
    
    return decorator

class HashableClass(ABC):
    @property
    @abstractmethod
    def _identifying_key(self):
        pass

    def __hash__(self):
        return hash(self._identifying_key)
    
    def __eq__(self, other: "HashableClass"):
        return self._identifying_key == other._identifying_key

"""
Results loading code
"""

@dataclass
class SteerResult(HashableClass):
    name: str
    test: Optional[datasets.Dataset] = None
    rand: Optional[datasets.Dataset] = None
    steer: Optional[datasets.Dataset] = None

    _num_proc: Optional[int] = None
    __hash__ = HashableClass.__hash__
    __eq__ = HashableClass.__eq__

    @property
    def _identifying_key(self) -> Tuple:
        return (self.name,bool(self.test),bool(self.rand),bool(self.steer))

    def __getitem__(self, key:str):
        return getattr(self,key)

    def set_num_proc(self,num_proc: int):
        self._num_proc = num_proc

    @property
    def subset(self) -> str:
        return self.name.name if isinstance(self.name, Path) else self.name
    
    @property
    def lang(self) -> str:
        lang = self.subset.split("-")[1]
        assert lang in ["py","ts"]
        return lang
    
    @property
    def mutations(self) -> str:
        muts = self.subset.split("-")[2]
        assert muts in ALL_MUTATIONS
        return muts
    
    @property
    def layers(self) -> str:
        return self.subset.split("-")[3]
    
    @classmethod
    def from_local(cls, path:Path) -> "SteerResult":
        datasets.disable_progress_bar()
        test,rand,steer=None,None,None
        if (path / "test-0-of-1.parquet").exists():
            test = datasets.Dataset.from_parquet((path / "test-0-of-1.parquet").as_posix())
        elif (path / "test_steering_results").exists():
            test = datasets.load_from_disk((path / "test_steering_results").as_posix())

        if (path / "rand-0-of-1.parquet").exists():
            rand = datasets.Dataset.from_parquet((path / "rand-0-of-1.parquet").as_posix())
        elif (path / "test_steering_results_rand").exists():
            rand = datasets.load_from_disk((path / "test_steering_results_rand").as_posix())
        
        if (path / "steer-0-of-1.parquet").exists():
            steer = datasets.Dataset.from_parquet((path / "steer-0-of-1.parquet").as_posix())
        elif (path / "steer_steering_results").exists():
            steer = datasets.load_from_disk((path / "steer_steering_results").as_posix())
        
        return cls(path,test=test,rand=rand,steer=steer)

    def missing_results(self) -> str:
        missing = []
        for split in ["test","rand","steer"]:
            if not getattr(self,split):
                missing.append(self.name + f"/{split}")
        if len(missing) == 3:
            return self.name
        return missing
    
    def to_success_dataframe(self) -> pd.DataFrame:
        name = self.subset
        num_layers = len(self.layers.split("_"))

        all_dfs = []
        for split in ["test","rand","steer"]:
            if self[split]:
                df = self[split].to_pandas()
                df = build_success_df(df)
                df.columns = [f"{split}_{c}" for c in df.columns]
                all_dfs.append(df)

        df = pd.concat(all_dfs, axis=1)
        df["lang"] = self.lang
        df["mutations"] = self.mutations
        df["layers"] = self.layers
        df["start_layer"] = int(self.layers.split("_")[0])
        df["model"] = get_model_name(name)
        df["num_layers"] = num_layers
        return df
    
    def to_errors_dataframe(self, split:str, disable_tqdm:bool=True) -> pd.DataFrame:
        num_proc = (self._num_proc or cpu_count())
        COLUMNS = ["steering_success","typechecks_before","typechecks_after", 
           "errors_before", "errors_after","fim_type","change","mutated_program"]
        ds = self[split].map(lambda x: 
                {**x, 
                "change": (x['mutated_generated_text'],x['steered_predictions']),
                "_before_steering_prog": x["mutated_program"].replace("<FILL>", x["mutated_generated_text"]),
                "_after_steering_prog":  x["mutated_program"].replace("<FILL>", x["steered_predictions"]),
                "steering_success": x["fim_type"] == x["steered_predictions"]
            }, num_proc=num_proc)
    
        # remove typechecks cols for safety, do not confuse results!
        ds = ds.remove_columns(["typechecks","errors"])
        
        # typecheck before/after steer
        result_before = multiproc_typecheck(ds, num_proc, lang=self.lang, 
                            colname="_before_steering_prog", disable_tqdm=disable_tqdm)
        result_after = multiproc_typecheck(ds, num_proc, lang=self.lang, 
                            colname="_after_steering_prog", disable_tqdm=disable_tqdm)
        before_df = pd.DataFrame.from_records(result_before).rename(
            columns={"typechecks":"typechecks_before","errors":"errors_before"})
        after_df = pd.DataFrame.from_records(result_after).rename(
            columns={"typechecks":"typechecks_after","errors":"errors_after"})
        
        # merge into one dataset
        df = pd.merge(before_df, after_df, on=["_before_steering_prog","_after_steering_prog"])
        # sanity check
        for col in ["steering_success","fim_type","mutated_program"]:
            assert list(df[f"{col}_x"]) == list(df[f"{col}_y"])
        
        df = df.rename(columns={
                "mutation": self.mutations,
                "steering_success_x":"steering_success", 
                "mutated_program_x":"mutated_program",
                "change_x": "change",
                "fim_type_x": "fim_type"
            })
        df = df[COLUMNS]
        
        for label in ["errors_before","errors_after"]:
            df[label] = df[label].apply(
                lambda x: remove_warnings(remove_filename(x, self.lang), self.lang) 
                    if isinstance(x, str) else str(x))
        return df

@dataclass
class ResultKeys(HashableClass):
    model:str
    lang:Optional[str]=None
    start_layer:Optional[int]=None
    mutation:Optional[str]=None
    interval:Optional[int]=None
    prefix:Optional[str]=None

    __hash__ = HashableClass.__hash__
    __eq__ = HashableClass.__eq__
    @property
    def _identifying_key(self) -> Tuple:
        return (self.model,self.lang,self.start_layer,self.mutation,self.interval,self.prefix)

    def __getitem__(self, key:str):
        return getattr(self,key)
    
    def get(self, key, other):
        if hasattr(self, key):
            return getattr(self, key)
        else:
            return other
    
    def expand(self) -> List[Tuple]:
        assert not self.interval or self.interval in [1,3,5]
        assert not self.start_layer or self.start_layer <= model_n_layer(self.model)-self.interval
        model = self.model
        lang = self.get("lang",None)
        start_layer = self.get("start_layer",None)
        mutation = self.get("mutation",None)
        interval = self.get("interval",None)
        prefix = self.get("prefix",None)
        members = []
        lang = [lang] if lang else ["py","ts"]
        start_layer = [int(start_layer)] if start_layer else list(range(model_n_layer(model)))
        mutation = [mutation] if mutation else ALL_MUTATIONS
        interval = [int(interval)] if interval else [1,3,5]
        prefix = prefix or ""
        for l in lang:
            for m in mutation:
                for s in start_layer:
                    for i in interval:
                        if int(s) > (model_n_layer(model) - i):
                            continue
                        members.append((model, l, m, s, i, prefix))
        return members

class ResultsLoader:
    """
    Performant load steering results either from local .parquet or from the hub.
    Will save a cached copy of hub data.

    Specify model, mutation, language, interval, start_layer, and optional split.
    Can specify num workers.
    Provides several functions for post processing data
    """
    local : bool
    hf_repo : str = "nuprl-staging/type-steering-results"
    auth_token: Optional[str] = None
    cache_dir: Optional[str] = None
    _prefetched: Set[Tuple[str]] = set()
    
    def __init__(self, 
        local: bool, 
        auth_token : Optional[str] = None, 
        cache_dir: Optional[str] = None
    ):
        self.local = local
        self.auth_token = (local or auth_token or os.environ["HF_AUTH_TOKEN"])
        if not cache_dir:
            cache_dir = "/tmp/codetrace_results_loader"
            if Path(cache_dir).exists():
                print(f"Loading from existing {cache_dir}")
            os.makedirs(cache_dir, exist_ok=True)
        print_color(f"Loading from local: {local}", "green")
        self.cache_dir = cache_dir

    def get_subset(
        self,
        model:str,
        lang:Optional[str]=None,
        mutation:Optional[str]=None,
        start_layer:Optional[int]=None,
        interval:Optional[int]=None,
        prefix:Optional[str]=None,
    ) -> str:
        layers = "_".join([str(start_layer+i) for i in range(interval)])
        return f"{prefix}steering-{lang}-{mutation}-{layers}-{model}"

    def prefetch(self, keys: ResultKeys):
        """
        Do prefetch
        """
        def all_splits(prefix: str) -> List[str]:
            return [prefix + f"/{split}-0-of-1.parquet" for split in ["test","rand","steer"]]

        expanded = keys.expand()
        all_files = []
        for e in expanded:
            all_files += all_splits(self.get_subset(*e))
        all_files = [f for f in all_files if not Path(f).exists()]
        cmd = [
            "huggingface-cli", "download", 
            "--repo-type", "dataset", 
            "--force-download", "--quiet",
            "--token", self.auth_token, 
            "--local-dir", self.cache_dir, 
            "nuprl-staging/type-steering-results", *all_files]
        
        subprocess.run(cmd)
        for r in expanded:
            self._prefetched.add(r)
    
    def is_prefetched(self, keys: ResultKeys) -> bool:
        return set(keys.expand()).issubset(self._prefetched)
    
    def load_data(self, keys: ResultKeys) -> List[SteerResult]:
        if not self.local and not self.is_prefetched(keys):
            self.prefetch(keys)
        return [SteerResult.from_local( Path(self.cache_dir) / self.get_subset(*e)) 
                for e in keys.expand()]
    

"""
Loading scripts
"""
def _to_success_dataframe(x:SteerResult):
    return x.to_success_dataframe()

def _missing_results(x: SteerResult):
    return x.missing_results()

def load_success_data(
    model:str, 
    num_proc:int,
    cache_dir: Optional[Path]=None,
    **keys,
) ->Tuple[pd.DataFrame,List[str]]:
    loader = ResultsLoader(Path(cache_dir).exists(), cache_dir=cache_dir)
    results = loader.load_data(ResultKeys(model=model, **keys))
    with Pool(num_proc) as p, tqdm(total=len(results)) as pbar:
        all_dfs = []
        for result in p.imap(_to_success_dataframe, results):
            pbar.update()
            pbar.refresh()
            all_dfs.append(result)
        missing_test_results = p.map(_missing_results, results)
    p.close()
    p.join()
    return pd.concat(all_dfs), list(it.chain(*missing_test_results))

def load_errors_data(
    model:str, 
    num_proc:int,
    split:str,
    cache_dir: Optional[Path]=None,
    **keys,
) ->Tuple[pd.DataFrame,List[str]]:
    loader = ResultsLoader(Path(cache_dir).exists(), cache_dir=cache_dir)
    results = loader.load_data(ResultKeys(model=model, **keys))
    all_dfs = []
    for r in tqdm(results, desc="Collecting errors with typechecker"):
        r.set_num_proc(num_proc)
        all_dfs.append(r.to_errors_dataframe(split))
    return pd.concat(all_dfs, axis=0)


"""
Tests
"""
def test_result_loader():
    cache_dir="/tmp/testing_result_loader"
    loader = ResultsLoader(False, cache_dir=cache_dir)
    keys = ResultKeys(**{"model":"qwen2p5_coder_7b_base","lang":"ts","interval":1, "start_layer":27})
    results = loader.load_data(keys)
    # print(results)
    for m in ALL_MUTATIONS:
        assert (Path(cache_dir) / f"steering-ts-{m}-27-qwen2p5_coder_7b_base").exists()
    
def test_load_data():
    df, missing = load_success_data("qwen2p5_coder_7b_base", 10, "/mnt/ssd/franlucc/scratch/type-steering-results")
    print(df)
    print(missing)
    assert len(df) > 0
    df, missing = load_success_data("qwen2p5_coder_7b_base", 10, None)
    print(df)
    print(missing)
    assert len(df) > 0

@cache_to_dir("/tmp/test_caching_codetrace")
def _save(item):
    return item

def test_caching():
    assert os.environ.get('PYTHONHASHSEED',None)=="42","Keep hash seed consistent for caching"
    steer_result = SteerResult("beebboop",None,None,True)
    result_keys = ResultKeys(model="model")
    steer_result2 = SteerResult("beebboop",None,None,True)
    result_keys2 = ResultKeys(model="model")

    assert result_keys == result_keys2
    assert steer_result == steer_result2
    _save(result_keys)
    _save(steer_result)
    _save(steer_result2)
    _save(result_keys2)
    assert len([*Path("/tmp/test_caching_codetrace").glob("*.pkl")]) == 2
