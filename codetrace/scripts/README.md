# Scripts

Scripts for running type inference activation steering

# Overview

`completions.py`: 
    given a dataset with `fim_program` and `fim_type` columns, filter by 1-token `fim_type` and
    produce 1-tok completions.

`fim_dataset.py`:
    takes a typescript program dataset and creates prompts for type annotation for each possible FIM in the program

`launch_steer.py`:
    Performs following tasks for steering.
        1. splits data into steering and test data
        2. takes steering split and computes steering tensor
        3. run steering on eval data with computed tensor

`mutate_dataset.py`:
    mutate a python or ts dataset with options `rename_types`, `rename_vars`, `delete_annotation`, then perform completions.

`typecheck_ds.py`:
    typecheck model completions