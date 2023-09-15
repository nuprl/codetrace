from datasets import *
import re
import sys

# py_data = load_dataset("bigcode/the-stack", split="train", data_files="data/python/train-00000-of-00206.parquet")
# java_data = load_dataset("bigcode/the-stack", split="train", data_files="data/java/train-00005-of-00285.parquet")

def crop_to_int_val(ex, shorten_to_n_lines=0, crop_before_int=True):
    """
    Takes a python code ex and crops to the first int value
    """
    if shorten_to_n_lines == 0:
        crop_to_int = ex
    else:
        crop_to_int = "\n".join(ex.split("\n")[:-shorten_to_n_lines])
    if crop_before_int:
        crop_to_int = re.findall(r"([\w\W]*?)\d", crop_to_int)[0]
    else:
        crop_to_int = re.findall(r"[\w\W]*?\d", crop_to_int)[0]
    return crop_to_int

def crop_to_int_var(ex, shorten_to_n_lines=0, crop_before_int=True):
    """
    Takes a python code ex and crops to the first int variable (in defn)
    """
    if shorten_to_n_lines == 0:
        crop_to_int = ex
    else:
        crop_to_int = "\n".join(ex.split("\n")[:-shorten_to_n_lines])
    var_name = re.findall(r"def [\W\w]*?\(([\W\w]*?): int", crop_to_int)[0]
    if crop_before_int:
        try:
            crop_to_int = re.findall(r"([\w\W]*)"+var_name, crop_to_int)[0]
        except:
            crop_to_int = None, None
    else:
        crop_to_int = re.findall(r"[\w\W]*"+var_name, crop_to_int)[0]
    return crop_to_int, var_name

def select_int_function(selected_data: Dataset):
    """
    takes dataset and extract a function with int in def
    """
    edited_data = {"train":[]}
    for ex in selected_data:
        # cut out function with int
        ints = re.findall(r'\n\n *def[\w\W]*?:[\w\W]*?\d[\w\W]*?\n', ex)
        unrolled = [i.split("\n\n")[-1] for i in ints]
        edited_data["train"] += unrolled
    return Dataset.from_dict(edited_data)

def select_short_py_func_with_asserts(dataset: Dataset, nlines=10, max_ex=sys.maxsize):
    """
    selects short python functions with asserts
    """
    edited_data = {"train":[]}
    for i,ex in enumerate(dataset):
        if i > max_ex:
            break
        assert(isinstance(ex, dict)), ex
        func = ex["content"]
        if len(func.split("\n")) < nlines:
            edited_data["train"].append(ex)
    return Dataset.from_dict(edited_data)

def crop_to_assert(ex: dict):
    """
    Note input ex is dict with field for tests (asserts) and content (func)
    """
    func = ex["content"]
    test = ex["tests"][0]
    assert test is not None, ex
    if "!=" in test:
        crop_assert = test.split("!=")[0] + "!= "
    elif "==" in test:
        crop_assert = test.split("==")[0] + "== "
    else:
        crop_assert = ""
    return func + "\n\n" + crop_assert
    