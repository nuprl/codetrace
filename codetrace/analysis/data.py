import pandas as pd
import datasets
from scipy import stats
from collections import namedtuple
from codetrace.fast_utils import batched_apply, make_batches
from typing import Dict,List,Tuple,Optional,TypedDict,Set
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass
from huggingface_hub import hf_hub_download
import os
import subprocess
import itertools as it
from multiprocessing import Pool
from codetrace.scripts.typecheck_ds import multiproc_typecheck
import re

MUTATIONS_RENAMED = {
    "types": "Rename types",
    "vars": "Rename variables",
    "delete": "Remove type annotations",
    "types_vars": "Rename types and variables",
    "types_delete": "Rename types and remove type annotations",
    "vars_delete": "Rename variables and remove type annotations",
    "delete_vars_types": "All edits",
}
ALL_MODELS = ["qwen2p5_coder_7b_base","CodeLlama-7b-Instruct-hf","starcoderbase-1b","starcoderbase-7b",
              "Llama-3.2-3B-Instruct"]

ALL_MUTATIONS = MUTATIONS_RENAMED.keys()

"""
Parsing layers, models, languages
"""

def remove_filename(text: str, lang_ext: str) -> str:
    return re.sub(r".*?\.{lang}".format(lang=lang_ext), "", text)
 
def remove_warnings(error: str, lang:str) -> str:
    if lang == "py":
        error_list = error.split("\n")
        return "\n".join([e.split("- error:")[-1].strip() for e 
                          in error_list if "- error:" in e])
    elif lang == "ts":
        error_list = re.split(r"(: error TS\d+)",error)
        return "\n".join([e.replace(": ","") for e in error_list if "error TS" in e])
    else:
        raise NotImplementedError(lang)
    
def get_model_name(name: str)->Optional[str]:
    for model in ALL_MODELS:
        if model in name.lower():
            return model
    raise ValueError(f"Name {model} not found")

def full_language_name(lang: str) ->str:
    if lang == "py":
        return "Python"
    elif lang == "ts":
        return "TypeScript"
    else:
        raise ValueError(f"Not found {lang}")

def get_ranges(num_layers: int, interval: int):
    for i in range(0, num_layers):
        if i + interval <= num_layers:
            yield "_".join(map(str, range(i, i + interval)))

def all_subsets(lang:str, model:str, n_layers:int, interval:int):
    subsets = []
    ranges = get_ranges(n_layers, interval)
    for r in ranges:
        for m in MUTATIONS_RENAMED.keys():
            subsets.append(f"steering-{lang}-{m}-{r}-{model}")
    return subsets

def model_n_layer(model: str) -> int:
    if "qwen" in model.lower():
        return 28
    elif "codellama" in model.lower():
        return 32
    elif "starcoderbase-1b" in model.lower():
        return 24
    elif "starcoderbase-7b" in model.lower():
        return 42
    elif "llama" in model.lower():
        return 28
    else:
        raise NotImplementedError(f"Model {model} model_n_layer not implemented!")

"""
Results loading code
"""

@dataclass
class SteerResult:
    name: str
    test: Optional[datasets.Dataset] = None
    rand: Optional[datasets.Dataset] = None
    steer: Optional[datasets.Dataset] = None

    def __getitem__(self, key:str):
        return getattr(self,key)

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
        
        return cls(path,test,rand,steer)

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
    
    def to_errors_dataframe(self, num_proc:int = 10) -> pd.DataFrame:
        COLUMNS = ["steering_success","typechecks_before","typechecks_after", 
           "errors_before", "errors_after","fim_type","change","program"]
        ds = ds.map(lambda x: 
                {**x, 
                "change": (x['mutated_generated_text'],x['steered_predictions']),
                "_before_steering_prog": x["mutated_program"].replace("<FILL>", x["mutated_generated_text"]),
                "_after_steering_prog":  x["mutated_program"].replace("<FILL>", x["steered_predictions"]),
                "steering_success": x["fim_type"] == x["steered_predictions"]
            }, num_proc=num_proc)
    
        # remove typechecks cols for safety
        ds = ds.remove_columns(["typechecks","errors"])
        
        # typecheck before/after steer
        result_before = multiproc_typecheck(ds, num_proc, lang=self.lang, colname="_before_steering_prog")
        result_after = multiproc_typecheck(ds, num_proc, lang=self.lang, colname="_after_steering_prog")
        before_df = pd.DataFrame.from_records(result_before).rename(
            columns={"typechecks":"typechecks_before","errors":"errors_before"})
        after_df = pd.DataFrame.from_records(result_after).rename(
            columns={"typechecks":"typechecks_after","errors":"errors_after"})
        
        # merge into one dataset
        df = pd.merge(before_df, after_df, 
                      on=["_before_steering_prog","_after_steering_prog","mutated_program"])
        assert list(df["steering_success_x"]) == list(df["steering_success_y"])
        assert list(df["fim_type_x"]) == list(df["fim_type_y"])
        df = df.rename(columns={
                "mutation": self.mutations,
                "steering_success_x":"steering_success", 
                "mutated_program_x":"program",
                "change_x": "change",
                "fim_type_x": "fim_type"
            })
        df = df[COLUMNS]
        
        for label in ["errors_before","errors_after"]:
            df[label] = df[label].apply(
                lambda x: remove_warnings(remove_filename(x, self.lang), self.lang) 
                        if isinstance(x, str) else "")
        return df

@dataclass
class ResultKwargs:
    model:str
    lang:Optional[str]=None
    start_layer:Optional[int]=None
    mutation:Optional[str]=None
    interval:Optional[int]=None
    prefix:Optional[str]=None

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
        self.auth_token = (auth_token or os.environ["HF_AUTH_TOKEN"])
        if not cache_dir:
            cache_dir = "/tmp/codetrace_results_loader"
            if Path(cache_dir).exists():
                print(f"Loading from existing {cache_dir}")
            os.makedirs(cache_dir, exist_ok=True)
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

    def prefetch(self, keys: ResultKwargs):
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
    
    def is_prefetched(self, keys: ResultKwargs) -> bool:
        return set(keys.expand()).issubset(self._prefetched)
    
    def load_data(self, keys: ResultKwargs) -> List[SteerResult]:
        if not self.local and not self.is_prefetched(keys):
            self.prefetch(keys)
        return [SteerResult.from_local( Path(self.cache_dir) / self.get_subset(*e)) 
                for e in keys.expand()]
        
def build_success_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a dataset with "steered_predictions" and "fim_type"
    build a dataset with success stats:
        ["num_succ", "tot_succ", "mean_succ"]
    """
    df["succ"] = df["steered_predictions"] == df["fim_type"]
    return pd.DataFrame.from_records([
        {"num_succ":df["succ"].sum(), 
         "tot_succ":df["succ"].count(),
         "mean_succ":df["succ"].mean(),}])

"""
Loading scripts
"""
def _to_success_dataframe(x:SteerResult):
    return x.to_success_dataframe()

def _missing_results(x: SteerResult):
    return x.missing_results()

def _to_errors_dataframe(x: SteerResult):
    return x.to_errors_dataframe()

def load_success_data(
    model:str, 
    num_proc:int,
    cache_dir: Optional[Path]=None,
    **keys,
) ->Tuple[pd.DataFrame,List[str]]:
    loader = ResultsLoader(Path(cache_dir).exists(), cache_dir=cache_dir)
    results = loader.load_data(ResultKwargs(model=model, **keys))
    with Pool(num_proc) as p, tqdm(total=len(results)) as pbar:
        all_dfs = []
        for result in p.imap(_to_success_dataframe, results):
            pbar.update()
            pbar.refresh()
            all_dfs.append(result)
        missing_test_results = p.map(_missing_results, results)
    
    return pd.concat(all_dfs), list(it.chain(*missing_test_results))

def load_errors_data(
    model:str, 
    num_proc:int,
    cache_dir: Optional[Path]=None,
    **keys,
) ->Tuple[pd.DataFrame,List[str]]:
    loader = ResultsLoader(Path(cache_dir).exists(), cache_dir=cache_dir)
    results = loader.load_data(ResultKwargs(model=model, **keys))
    with Pool(num_proc) as p, tqdm(total=len(results)) as pbar:
        all_dfs = []
        for result in p.imap(_to_errors_dataframe, results):
            pbar.update()
            pbar.refresh()
            all_dfs.append(result)
    
    return pd.concat(all_dfs, axis=0)

"""
Metrics
"""
@dataclass
class CondProb:
    prob_a : float
    prob_b : float
    prob_a_and_b : float
    prob_a_given_b : float
    num_a : int
    num_b : int
    total : int
    label_a : str
    label_b : str

    def __repr__(self):
        probs_str = f"""P({self.label_a} | {self.label_b}) = P({self.prob_a_and_b}) / P({self.prob_b}) 
        = {self.prob_a_given_b}"""
        return f"[{self.total} samples] {probs_str}"
    
def conditional_prob(var_a: str, var_b: str, df: pd.DataFrame) -> CondProb:
    """
    probability of A|B
    
    P(A|B) = P(AnB)/P(B)
    
    """
    prob_a = df[var_a].mean()
    prob_b = df[var_b].mean()
    prob_anb = (df[var_a] & df[var_b]).mean()
    prob_a_given_b = 0 if prob_anb == 0 else prob_anb / prob_b
    return CondProb(prob_a, prob_b, prob_anb, prob_a_given_b, 
                    df[var_a].sum(), df[var_b].sum(), len(df), var_a, var_b)

def correlation(df: pd.DataFrame, var_a:str, var_b:str):
    return stats.pearsonr(df[var_a], df[var_b])

"""
Tests
"""
def test_result_loader():
    cache_dir="/tmp/testing_result_loader"
    loader = ResultsLoader(False, cache_dir=cache_dir)
    keys = ResultKwargs(**{"model":"qwen2p5_coder_7b_base","lang":"ts","interval":1, "start_layer":27})
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


if __name__ == "__main__":
    test_result_loader()
    test_load_data()