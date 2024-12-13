import datasets
import argparse
import pandas as pd
from codetrace.analysis.data import ResultsLoader,ResultKeys
from tqdm import tqdm
from codetrace.analysis.utils import ALL_MODELS,ALL_MUTATIONS,MUTATIONS_RENAMED
from pathlib import Path
import os

def compare_icl(steering_df:pd.DataFrame, icl_df:pd.DataFrame, outfile:str) -> pd.DataFrame:
    print(steering_df.columns, icl_df.columns)
    icl_df["type"] = "icl"
    steering_df["type"] = "steer"
    df = pd.concat([icl_df,steering_df], axis=0)
    df = df[["model","mutations","test_mean_succ","lang","type"]]
    df = df.sort_values(["lang","model","mutations","type"])
    df.to_csv(outfile)
    return df

def plot_icl(df:pd.DataFrame, outfile:str):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=str)
    parser.add_argument("outfile", type=str)
    parser.add_argument("--num-proc", type=int, default=40)
    parser.add_argument("--interval", choices=[1,3,5], type=int, default=5)
    assert os.environ.get('PYTHONHASHSEED',None)=="42",\
        "Set PYTHONHASHSEED to 42 for consistent and reliable caching"
    args = parser.parse_args()
    
    all_results = []
    loader = ResultsLoader(Path(args.results_dir).exists(), 
                           cache_dir=args.results_dir)
    for model in tqdm(ALL_MODELS, desc="models"):
        keys = ResultKeys(model=model, interval=args.interval)
        results = loader.load_data(keys)
        all_results += results
    
    processed_results = []
    for r in tqdm(all_results, "checking"):
        try:
            assert r.test != None, r.name
        except:
            if "ts-types-" in r.name and "starcoderbase-7b" in r.name:
                continue
        rdf = r.to_dataframe("test")
        rdf = rdf.groupby(["mutations","lang","model","start_layer"]).agg(
            {"test_is_success":"mean"}).reset_index()
        processed_results.append(rdf)

    df = pd.concat(processed_results, axis=0)
    df = df.rename(columns={"test_is_success":"test_mean_succ"})
    print(df.columns)
    df = df.groupby(["model","mutations","lang"]).agg({"test_mean_succ":"max"}).reset_index()

    # collect icl df
    all_icl_dfs = []
    for model in ALL_MODELS:
        for muts in ALL_MUTATIONS:
            for lang in ["py","ts"]:
                try:
                    icl_df = datasets.load_from_disk(f"results/icl_{model}-{lang}-{muts}").to_pandas()
                    icl_df["mutations"] = muts
                    icl_df["model"] = model
                    icl_df["lang"] = lang
                except:
                    if "starcoder-7b-ts-types" in f"{model}-{lang}-{muts}":
                        continue
                all_icl_dfs.append(icl_df)
    df_icl = pd.concat(all_icl_dfs, axis=0).reset_index()
    df_icl = df_icl.groupby(["model","mutations","lang"]).agg({"correct":"mean"}).reset_index()
    print(df_icl.columns)
    df_icl = df_icl.rename(columns={"correct":"test_mean_succ"})
    df = compare_icl(df, df_icl, args.outfile)
    plot_icl(df, args.outfile.replace(".csv",".pdf"))
