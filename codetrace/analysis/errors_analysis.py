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
import os
import csv
from tqdm import tqdm
import argparse
from typing import Optional,List,Dict,Any,Union
from codetrace.utils import load_dataset, print_color
import pandas as pd
from pathlib import Path
from ruamel.yaml import YAML
from codetrace.analysis.utils import conditional_prob, correlation
from codetrace.analysis.utils import ALL_MUTATIONS, correlation, ANALYSIS_CACHE_DIR, ANALYSIS_FIGURE_DIR
from codetrace.analysis.data import SteerResult, build_success_data,ResultKeys, ResultsLoader,cache_to_dir

yaml = YAML()
yaml.default_flow_style = False
yaml.sort_keys = False
yaml.default_style = "|"

def log_to_csv(data: Union[Dict[str,Any],List[Dict[str,Any]]], fname:str):
    results_csv = Path(ANALYSIS_FIGURE_DIR) / fname
    if results_csv.exists():
        mode,do_header = "a",False
    else:
        mode,do_header = "w",True
    
    with open(results_csv, mode) as fp:
        keys = (data[0] if isinstance(data, list) else data).keys()
        writer = csv.DictWriter(fp, fieldnames=keys)
        if do_header:
            writer.writeheader()
        
        if isinstance(data, list):
            writer.writerows(data)
        else:
            writer.writerow(data)

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
        df = SteerResult.from_local(name=results_dataset, **{split: ds}).to_errors_dataframe(num_proc=num_proc)
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

@cache_to_dir(ANALYSIS_CACHE_DIR)
def cached_to_errors_df(result: SteerResult, split:str)->List[SteerResult]:
    return result.to_errors_dataframe(split, True)

def RQ1(model:str, result_path:Path, lang:str, interval:int, num_proc:int) -> int:
    """
    lemma: "Activation steering identifies error subspaces"
    corollary: "that's why lang transfer works, subspaces in common between langs"

    Given a model, lang, mut and MAX SUCCESS RATE layer

    1. when num `typechecks before` is high, steering success is low.
    2. the fim type is a subtype of gold fim_type in the cases where steering success is low
    """
    all_results = []
    edfs = []
    steering_success_x,typechecks_before_y = [],[]
    loader = ResultsLoader(Path(result_path).exists(), cache_dir=result_path)
    for m in tqdm(ALL_MUTATIONS, desc="Processing muts"):
        # load all data
        keys = ResultKeys(model=model, lang=lang, interval=interval, mutation=m)
        print(keys)
        results = loader.load_data(keys)

        # get most performant layer
        try:
            mut_succ_df,_ = build_success_data(results, num_proc)
        except:
            print_color(f"Failed to collect {m}", "red")
            steering_success_x.append(-1)
            typechecks_before_y.append(-1)
            continue
        
        # load best layer data
        best_layer = mut_succ_df["start_layer"].iloc[mut_succ_df["test_mean_succ"].idxmax()]

        results = loader.load_data(ResultKeys(**{**keys.to_dict(), "start_layer":best_layer}))
        all_results += results
        for idx,r in enumerate(results):
            r.set_num_proc(num_proc)
            results[idx] = cached_to_errors_df(r, "test")
            
        edf = pd.concat(results, axis=0)
        edf["mutation"] = m
        edfs.append(edf)
        cprob = conditional_prob("steering_success","typechecks_before", edf)
        print(cprob)
        steering_success_x.append(cprob.prob_a)
        typechecks_before_y.append(cprob.prob_b)
        
    all_error_dfs = pd.concat(edfs, axis=0)

    # 1. when num `typechecks before` is high, steering success is low. Expect an inverse correlation
    df_muts = all_error_dfs.groupby("mutation").agg({"steering_success":"mean", "typechecks_before": "sum"})
    print("Correlation typechecks_before and steering success: ", \
            correlation(df_muts, "typechecks_before", "steering_success"))
    
    cprob = conditional_prob("steering_success", "typechecks_before", all_error_dfs)
    print(cprob)

    data = {
        "model":model, 
        "prob_steering_success":steering_success_x, 
        "prob_typechecks_before":typechecks_before_y, 
        "lang":lang,
        "mutations":ALL_MUTATIONS,
        "interval":interval
    }
    
    log_to_csv(data, "rq1.csv")

    # 2. the fim type is a subtype of gold fim_type in the cases where steering success is low
    # simply save type before, type after, subset data, success_data
    data = all_error_dfs.copy()
    data = data[["change","fim_type","steering_success","mutation","typechecks_before","typechecks_after"]]
    data["model"] = model
    data["lang"] = lang
    data["interval"] = interval
    log_to_csv(data.to_dict("records"), "rq1.2.csv")


if __name__ == "__main__":
    assert os.environ.get('PYTHONHASHSEED',None)=="42",\
        "Set PYTHONHASHSEED to 42 for consistent and reliable caching"
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="command")

    subparser_rq1 = subparsers.add_parser("RQ1")
    subparser_rq1.add_argument("--model", required=True)
    subparser_rq1.add_argument("--result-path", required=True, type=Path, help="Can be local or hub subset")
    subparser_rq1.add_argument("--lang", required=True, choices=["py","ts"])
    subparser_rq1.add_argument("--interval", required=True, type=int, choices=[1,3,5])
    subparser_rq1.add_argument("--num-proc", type=int, default=40)

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

