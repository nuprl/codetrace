import pandas as pd
from scipy import stats
from pathlib import Path
from dataclasses import dataclass
from typing import List,Union
import re
import os
from typing import Optional

CWD=Path(__file__).parent.absolute()
ANALYSIS_CACHE_DIR=(CWD / "cache").as_posix()
ANALYSIS_FIGURE_DIR=(CWD / "figures").as_posix()
os.makedirs(ANALYSIS_CACHE_DIR, exist_ok=True)
os.makedirs(ANALYSIS_FIGURE_DIR, exist_ok=True)

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

ALL_MUTATIONS = sorted(MUTATIONS_RENAMED.keys())

"""
Parsing layers, models, languages
"""

def get_unique_value(df: pd.DataFrame, colname: str, n:int) -> Union[str, List[str]]:
    values = df[colname].unique()
    assert len(values) == n
    if n == 1:
        return values[0]
    else:
        return values

def remove_filename(text: str, lang_ext: str) -> str:
    # filename is hash
    return re.sub(r"\d+?\.{lang}".format(lang=lang_ext), "", text)
 
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
        if model.lower() in name.lower():
            return model
    raise ValueError(f"Name {name} not found")

def full_language_name(lang: str) ->str:
    if lang == "py":
        return "Python"
    elif lang == "ts":
        return "TypeScript"
    else:
        raise ValueError(f"Not found {lang}")

def full_model_name(model: str) -> str:
    if "qwen" in model.lower():
        return "Qwen 2.5 Coder 7B"
    elif "codellama" in model.lower():
        return "CodeLlama Instruct 7B"
    elif "starcoderbase-1b" in model.lower():
        return "StarcoderBase 1B"
    elif "starcoderbase-7b" in model.lower():
        return "StarcoderBase 7B"
    elif "llama" in model.lower():
        return "Llama 3.2 Instruct 3B"
    else:
        raise NotImplementedError(f"Model {model} model_n_layer not implemented!")
    
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
        label = f"P({self.label_a} | {self.label_b})"
        probs_str = f"{label} = P({self.prob_a_and_b:.2f}) / P({self.prob_b:.2f}) = {self.prob_a_given_b:.2f}"
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

def test_remove_warnings():
    error = '''\n:29:9 - information: Analysis of function \"validate\" is skipped\
\ because it is unannotated\n:75:9 - information: Analysis of function \"serialize\"\
\ is skipped because it is unannotated\n:103:9 - information: Analysis of function\
\ \"__len__\" is skipped because it is unannotated\n:106:9 - information: Analysis\
\ of function \"__setitem__\" is skipped because it is unannotated\n:121:13
\ - error: Module cannot be used as a type (reportGeneralTypeIssues)\n:133:9 - information:\
\ Analysis of function \"validate\" is skipped because it is unannotated\n1 error,\
\ 0 warnings, 5 informations \nWARNING: there is a new pyright version available\
\ (v1.1.388 -> v1.1.390).\nPlease install the new version or set PYRIGHT_PYTHON_FORCE_VERSION\
\ to `latest`\n\n\n
'''
    output = remove_warnings(error, "py")
    expected = 'Module cannot be used as a type (reportGeneralTypeIssues)'
    assert output.strip() == expected.strip()

if __name__ == "__main__":
    # call all test fns
    for key, value in globals().items():
        if callable(value) and key.startswith('test_'):
            value()