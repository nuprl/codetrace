import datasets
import glob
import pandas as pd

dataset_list = []
for path in glob.glob("*/*.ts"):
    content = open(path, "r").read()
    dataset_list.append({"content": content, "path": path})

# dataset from a list of dicts
df = pd.DataFrame(data=dataset_list)
dataset = datasets.Dataset.from_pandas(df)

dataset.save_to_disk("../ts-benchmark-dataset")

