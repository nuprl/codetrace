from codetrace.py_mutator import (
    add_type_aliases_after_imports as py_add_aliases,
    postprocess_type_annotation as py_postproc_annotation,
    postprocess_return_type as py_postproc_return_type,
    random_mutate as _py_mutate,
    mutate_captures as py_mutate_captures,
    apply_mutations as py_apply_mutations,
    find_mutation_locations as py_find_mut_locs
)
from codetrace.ts_mutator import (
    random_mutate as _ts_mutate,
    apply_mutations as ts_apply_mutations,
)
from functools import partial
py_mutate = partial(_py_mutate, debug_seed=-1)
ts_mutate = partial(_ts_mutate, debug_seed=-1)