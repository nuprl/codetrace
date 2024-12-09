"""
Given a dataset of results, find the typechecking errors before and
after steering model predictions. Show the errors that were corrected 
and those that weren't

RQs for error analysis:

lemma: "Activation steering identifies error subspaces"
corollary: "that's why lang transfer works, subspaces in common between langs"

Given a model, lang, mut and MAX SUCCESS RATE layer

1. when num `typechecks before` is high, steering success is low.
    + the fim type is a subtype of fim_type 

"""
from codetrace.scripts.typecheck_ds import multiproc_typecheck
import itertools as it
import argparse
from typing import Optional, List
import datasets
from codetrace.utils import load_dataset
import pandas as pd
from pathlib import Path
from collections import Counter
from ruamel.yaml import YAML
import numpy as np
import re
from codetrace.analysis.data import (
    conditional_prob, ALL_MUTATIONS, correlation, SteerResult, load_success_data,load_errors_data
)

yaml = YAML()
yaml.default_flow_style = False
yaml.sort_keys = False
yaml.default_style = "|"

def analyze_steering_effect_on_errors(
    results_dataset: str, 
    yamlout: str,
    num_proc: int = 10,
    split: Optional[str] = None,
    subset: Optional[str] = None,
):
    if Path(yamlout).exists():
        print(f"Skipping typechecking, resuming from saved {yamlout}")
        with open(yamlout, "r") as fp:
            data = yaml.load(fp)
        df = pd.DataFrame.from_records(data)
    else:
        ds = load_dataset(results_dataset, split=split, name=subset, trust_remote_code=True)
        df = SteerResult(name=results_dataset, **{split: ds}).to_errors_dataframe(num_proc=num_proc)
        assert df["typechecks_before"].value_counts()[False] > 0
        print(df["typechecks_before"].value_counts())
        data = df.to_dict(orient="records")
        data.sort(key=lambda x: x["steering_success"], reverse=True)
        with open(yamlout, "w") as fp:
            yaml.dump(data, fp)
    
    # # analyze
    # """
    # Get all steering_success errors before
    # """
    # df_succ = df[df["steering_success"]]
    # errors_solved = df_succ["errors_before"].apply(lambda x: x.split("\n"))
    
    # """
    # Get all steering_fail errors after
    # """
    # df_fail = df[~df["steering_success"]]
    # errors_failed = df_fail["errors_after"].apply(lambda x: x.split("\n"))
    # # errors_failed = [extract_error_msg(x, lang) for x in errors_failed if isinstance(x, str)]
    # # errors_solved = [extract_error_msg(x, lang) for x in errors_solved if isinstance(x, str)]
    # errors_failed = list(it.chain(*[e for e in errors_failed.to_list() if e != [""]]))
    # errors_solved = list(it.chain(*[e for e in errors_solved.to_list() if e != [""]]))

    # # errors solved should not be in errors failed
    # intersection = set(errors_failed).intersection(set(errors_solved))

    # print("COUNTER SOLVED", Counter(errors_solved), end="\n\n")
    # print("COUNTER FAILED",Counter(errors_failed), end="\n\n")
    
    # print("SUCC DID NOT SOLVE SOME OF THESE ERRORS", intersection, end="\n\n")

    # print("SUCC SOLVED THESE ERRORS", set(errors_solved).difference(set(errors_failed)), end="\n\n")

    # print("SUCC DID NOT SOLVE THESE ERRORS", set(errors_failed).difference(set(errors_solved)))

def RQ1(model:str, results:Path, lang:str, interval:int, num_proc:int=40) -> int:
    """
    lemma: "Activation steering identifies error subspaces"
    corollary: "that's why lang transfer works, subspaces in common between langs"

    Given a model, lang, mut and MAX SUCCESS RATE layer

    1. when num `typechecks before` is high, steering success is low.
        + the fim type is a subtype of fim_type 
    """
    df_success,_ = load_success_data(model, num_proc, results, lang=lang, interval=interval)
    _df = df_success.groupby("mutations").agg({"test_mean_succ":"max","start_layer":"first"}
                                              ).to_dict(orient="list")
    muts_to_best_layer = {k:v for (k,v) in zip(_df["mutations"], _df["start_layer"])}
    
    all_error_dfs = []
    for m in ALL_MUTATIONS:
        edf = load_errors_data(
                model, num_proc, results, mutation=m, 
                start_layer=muts_to_best_layer[m], 
                lang=lang, interval=interval
            )
        all_error_dfs.append(edf)
    all_error_dfs = pd.concat(all_error_dfs, axis=0)
    
    # 1. when num `typechecks before` is high, steering success is low.
    df_muts = all_error_dfs.groupby("mutation").agg({"steering_success":"mean", "typechecks_before": "sum"})
    print("Correlation typechecks_before and steering success: ", \
            correlation(df_muts, "typechecks_before", "steering_success"))
    
    cprob = conditional_prob("steering_success", "typechecks_before", df_muts)
    print(cprob)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="command")

    subparser_rq1 = subparsers.add_parser("RQ1")
    subparser_rq1.add_argument("--model", required=True)
    subparser_rq1.add_argument("--results", required=True, type=Path)
    subparser_rq1.add_argument("--lang", required=True, choices=["py","ts"])
    subparser_rq1.add_argument("--interval", required=True, type=int, choices=[1,3,5])

    args = parser.parse_args()
    args = vars(args)
    if args.pop("command") == "RQ1":
        RQ1(**args)

    # parser.add_argument("results_dataset", type=str)
    # parser.add_argument("yamlout", type=str)
    # # typechecker args
    # parser.add_argument("--lang", choices=["py", "ts"], required=True)

    # # data loading args
    # parser.add_argument("--split", type=str, default=None)
    # parser.add_argument("--subset", type=str, default=None)
    # parser.add_argument("--num-proc", type=int, default=10)

    # # optional: analyze errors in steering split
    # # parser.add_argument("results_dataset", type=str)

    # args = parser.parse_args()
    # main_with_args(**vars(args))


"""
PYTESTS
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
