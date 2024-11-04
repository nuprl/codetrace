# Scripts

Scripts for running type inference activation steering

# Overview

`completions.py`: 
    given a dataset with `fim_program` and `fim_type` columns, filter by 1-token `fim_type` and
    produce 1-tok completions.
`launch_steer.py`:
    Performs following tasks for steering.
        1. `make_steering_data_splits`: splits data into steering and eval data
        2. `make_steering_tensor`: takes steering split and computes steering tensor
        3. `run_steering`: run steering on eval data with computed tensor
        4. `layer_ablation`: perform steering at every layer window

`pipeline_steering.py`: 
    Performs all the steering tasks 1-4.

`make_fim_dataset.py`:
    takes a typescript program dataset and creates prompts for type annotation for each possible FIM in the program

`py_mutate_ds.py`:
    mutate a py dataset with options `rename_types`, `rename_vars`, `delete_annotation`, then perform completions.

`ts_mutate_ds.py`:
    mutate a ts dataset with options `rename_types`, `rename_vars`, `delete_annotation`, then perform completions.

`typecheck_ds.py`:
    typecheck model completions

`pipeline_mutations.py`:
    for a source dataset, perform all mutations and completions from combinations of  `rename_types`, `rename_vars`, `delete_annotation`