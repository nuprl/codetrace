# Codetrace

# Overview

- `./scripts`: python scripts for running the code. See `./scripts/README`

- `./bin`: bash/sbatch scripts for launching jobs on clusters

- `./analysis`: scripts for producing figures/analysing results

Main code:

- `base_mutator.py`: base class from which all mutators inherit. Based on `tree-sitter`.

- `py_mutator.py`: python mutator for performing semantics-preserving edits

- `ts_mutator.py`: typescript mutator for performing semantics-preserving edits

- `batched_utils.py`: utils for batched running of prompts

- `parsing_utils.py`: utils for forming FIM prompts

- `interp_utils.py`: utils for interp functions in `nnsight`

- `fast_utils.py`: parallelized data processing functions

- `steering.py`: steering code class

- `utils.py`: torch utils

- `vllm_utils.py`: vllm loading and completions functions
