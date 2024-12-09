"""
Given a dataset of results, find the typechecking errors before and
after steering model predictions. Show the errors that were corrected 
and those that weren't
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
yaml = YAML()
yaml.default_flow_style = False
yaml.sort_keys = False
yaml.default_style = "|"

COLUMNS = ["steering_success","typechecks_before","typechecks_after", 
           "errors_before", "errors_after","fim_type","change","program"]
SEP="<<"

def remove_filename(error: str, lang: str) -> str:
    return re.sub(r".*?\.{lang}".format(lang=lang), "", error)
 
def remove_warnings(error: str, lang:str) -> str:
    if lang == "py":
        error_list = error.split("\n")
        return "\n".join([e.split("- error:")[-1].strip() for e 
                          in error_list if "- error:" in e])
    elif lang == "ts":
        error_list = re.split(r"(: error TS\d+)",error)
        return "\n".join([e.replace(": ","") for e in error_list if "error TS" in e])
    else:
        raise NotImplementedError()

def build_error_dataset(
    results_dataset:str,
    lang:str,
    split:Optional[str]= None,
    subset:Optional[str]=None,
    num_proc:int=10
)-> pd.DataFrame:
    ds = load_dataset(results_dataset, split=split, name=subset, trust_remote_code=True)
    print(ds)
    assert SEP not in "\n".join(ds["mutated_program"]), "Pick another placeholder!"

    ds = ds.map(lambda x: {**x, 
                "change": f"{SEP}{x['mutated_generated_text']}{SEP}{x['steered_predictions']}{SEP}",
                "_before_steering_prog": x["mutated_program"].replace("<FILL>", x["mutated_generated_text"]),
                "_after_steering_prog":  x["mutated_program"].replace("<FILL>", x["steered_predictions"]),
                "changed_program": x["mutated_program"].replace(
                    "<FILL>", f"{SEP}{x['mutated_generated_text']}{SEP}{x['steered_predictions']}{SEP}"),
                "steering_success": x["fim_type"] == x["steered_predictions"]},
                num_proc=num_proc)
    
    # remove typechecks cols for safety
    ds = ds.remove_columns(["typechecks","errors"])
    print("Success rate steering:", ds.to_pandas()["steering_success"].mean())

    # typecheck before/after steer
    result_before = multiproc_typecheck(ds, num_proc, lang=lang, colname="_before_steering_prog")
    result_after = multiproc_typecheck(ds, num_proc, lang=lang, colname="_after_steering_prog")
    before_df = pd.DataFrame.from_records(result_before).rename(
        columns={"typechecks":"typechecks_before","errors":"errors_before"})
    after_df = pd.DataFrame.from_records(result_after).rename(
        columns={"typechecks":"typechecks_after","errors":"errors_after"})
    
    # merge into one dataset
    df = pd.merge(before_df, after_df, on=["_before_steering_prog","_after_steering_prog","mutated_program"])
    assert list(df["steering_success_x"]) == list(df["steering_success_y"])
    assert list(df["fim_type_x"]) == list(df["fim_type_y"])
    df = df.rename(columns={"steering_success_x":"steering_success", "changed_program_x":"program",
                            "change_x": "change",
                            "fim_type_x": "fim_type"})
    df = df[COLUMNS]
    
    for label in ["errors_before","errors_after"]:
        df[label] = df[label].apply(lambda x: remove_warnings(remove_filename(x, lang), lang) 
                                    if isinstance(x, str) else "")
    return df

# def extract_error_msg(e:str, lang:str) -> List[str]:
#     if lang == "ts":
#         raise NotImplementedError()
#     elif lang == "py":
#         return py_extract_error_msg(e)
#     else:
#         raise ValueError(f"Language {lang} not supported.")

# def py_extract_error_msg(e:str) -> List[str]:
#     error_list = e.split("\n")
#     for i in range(len(error_list)):
#         err = error_list[i].split("-")[-1]
#         if "(" and ")" in err:
#             # pyright error type like "(reportInvalidTypeForm)"
#             err = "error: " + err.split("(")[-1].replace(")","")
#         error_list[i] = err
#     return [err for err in error_list if "error:" in err]

def main_with_args(
    results_dataset: str, 
    lang: str,
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
        df = build_error_dataset(results_dataset, lang, split, subset, num_proc)
        assert df["typechecks_before"].value_counts()[False] > 0
        print(df["typechecks_before"].value_counts())
        data = df[COLUMNS].to_dict(orient="records")
        data.sort(key=lambda x: x["steering_success"], reverse=True)
        with open(yamlout, "w") as fp:
            yaml.dump(data, fp)
    
    # analyze
    """
    Get all steering_success errors before
    """
    df_succ = df[df["steering_success"]]
    errors_solved = df_succ["errors_before"].apply(lambda x: x.split("\n"))
    
    """
    Get all steering_fail errors after
    """
    df_fail = df[~df["steering_success"]]
    errors_failed = df_fail["errors_after"].apply(lambda x: x.split("\n"))
    # errors_failed = [extract_error_msg(x, lang) for x in errors_failed if isinstance(x, str)]
    # errors_solved = [extract_error_msg(x, lang) for x in errors_solved if isinstance(x, str)]
    errors_failed = list(it.chain(*[e for e in errors_failed.to_list() if e != [""]]))
    errors_solved = list(it.chain(*[e for e in errors_solved.to_list() if e != [""]]))

    # errors solved should not be in errors failed
    intersection = set(errors_failed).intersection(set(errors_solved))

    print("COUNTER SOLVED", Counter(errors_solved), end="\n\n")
    print("COUNTER FAILED",Counter(errors_failed), end="\n\n")
    
    print("SUCC DID NOT SOLVE SOME OF THESE ERRORS", intersection, end="\n\n")

    print("SUCC SOLVED THESE ERRORS", set(errors_solved).difference(set(errors_failed)), end="\n\n")

    print("SUCC DID NOT SOLVE THESE ERRORS", set(errors_failed).difference(set(errors_solved)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dataset", type=str)
    parser.add_argument("yamlout", type=str)
    # typechecker args
    parser.add_argument("--lang", choices=["py", "ts"], required=True)

    # data loading args
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--subset", type=str, default=None)
    parser.add_argument("--num-proc", type=int, default=10)

    # optional: analyze errors in steering split
    # parser.add_argument("results_dataset", type=str)

    args = parser.parse_args()
    main_with_args(**vars(args))


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