"""
Opens every .json.gz file in a directory and checks that completions[0].text
is equal to extras.canonical_solution. Results are grouped by extras.language.

Computes the accuracy of the model by langauge.
"""
import argparse
import json
import gzip
from pathlib import Path
import pandas as pd

pd.set_option("display.float_format", "{:.3f}".format)

def gunzip_file(path: Path):
    with gzip.open(path, "rt") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dirs", type=str, nargs="+", help="One or more directories containing results")
    args = parser.parse_args()

    # Create list to store results
    results = []
    
    for input_dir in args.input_dirs:
        for path in Path(input_dir).glob("*.json.gz"):
            data = gunzip_file(path)
            language = data["extras"]["language"]
            
            for completion in data["completions"]:
                is_correct = completion["text"].strip() == data["extras"]["canonical_solution"].strip()
                results.append({
                    "model": Path(input_dir).name,
                    "language": language,
                    "correct": is_correct
                })

    # Convert to DataFrame and compute statistics
    df = pd.DataFrame(results)
    
    # Group by model and language to compute accuracy
    language_stats = df.groupby(["model", "language"]).agg({
        "correct": ["count", "sum"]
    })

    # Get total samples per language for each model
    lang_totals = df.groupby(["model", "language"])["correct"].count().unstack()
    
    # Check if all models have same number of samples per language
    if not all(lang_totals.iloc[0].equals(row) for _, row in lang_totals.iterrows()):
        print("\nWARNING: Models have different numbers of samples per language:")
        print(lang_totals)
        print("\n")
    
    language_stats.columns = ["total", "correct"] 
    language_stats["accuracy"] = language_stats["correct"] / language_stats["total"]
    language_stats = language_stats.drop(columns=["total", "correct"])
    print(language_stats)


if __name__ == "__main__":
    main()
