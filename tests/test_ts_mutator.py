from codetrace.type_inf_exp.ts_mutator import *
from codetrace.utils import *
from pathlib import Path

parent_dir = Path(__file__).parent
test_files_dir = f"{parent_dir}/test_files"
with open(f"{test_files_dir}/ts_mutator_prog.ts") as f:
    program = f.read()
fim_type = "any"

DEBUG_SEED = -1

def write_results(out, mutations, file_name):
    with open(f"{test_files_dir}/{file_name}", "w") as f:
        f.write(out.replace("<FILL>", "any"))
    with open(f"{test_files_dir}/log_{file_name}", "w") as f:
        f.write("/*")
        for m in mutations:
            f.write(f"\n{m}\n")
        f.write("*/")
               
def test_mutate_rename_vars():
    """
    test just rename vars
    """
    out, mutations = random_mutate(program, fim_type, [mutation_rename_vars], debug_seed=DEBUG_SEED)
    write_results(out, mutations, "OUT_prog_only_rename_vars.ts")
    print(f"[ONLY] Num of renamed vars: {len(mutations)}")

def test_mutate_rename_types():
    """
    test just rename types
    """
    out, mutations = random_mutate(program, fim_type, [mutation_rename_type], debug_seed=DEBUG_SEED)
    write_results(out, mutations, "OUT_prog_only_rename_types.ts")
    print(f"[ONLY] Num of renamed types: {len(mutations)}")

def test_mutate_remove_annotations():
    """
    test just remove type annotations
    """
    out, mutations = random_mutate(program, fim_type, [mutation_delete_annotation], debug_seed=DEBUG_SEED)
    write_results(out, mutations, "OUT_prog_only_delete_types.ts")
    print(f"[ONLY] Num of removed annotations: {len(mutations)}")
    
def test_mutate_all_renames():
    """
    test all renames
    """
    out, mutations = random_mutate(program, fim_type, [mutation_rename_vars, mutation_rename_type], debug_seed=DEBUG_SEED)
    write_results(out, mutations, "OUT_prog_all_renames.ts")
    print(f"[ALL] Num of renamed vars+types: {len(mutations)}")
    
if __name__ == "__main__":
    print("Running tests")
    test_mutate_rename_vars()
    test_mutate_rename_types()
    test_mutate_remove_annotations()
    test_mutate_all_renames()