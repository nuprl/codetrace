import sys
import datasets
from codetrace.utils import print_color
from codetrace.scripts.typecheck_ds import multiproc_typecheck
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

if __name__ == "__main__":
    path = sys.argv[1]

    ds = datasets.Dataset.from_parquet(f"{path}/test-0-of-1.parquet")
    lang = path.split("steering-")[-1].split("-")[0]
    assert lang in ["py","ts"]
    ds_typechecked = multiproc_typecheck(list(ds), 40, lang=lang, colname="mutated_program")
    df = pd.DataFrame.from_records(ds_typechecked)
    if df["typechecks"].mean() != 1.0:
        print_color(f'{df["typechecks"].value_counts()}, {df["typechecks"].mean()}',"red")
        assert False

