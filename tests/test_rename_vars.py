from codetrace.type_inf_exp.scripts.rename_vars import *
from transformers import AutoTokenizer
import difflib

def test_rename_vars():
    program = """
export function buildInClause<T>(
  items,
  startingAt = 1,
) {
  return {
    in: items.map((_, i) => `$${i + startingAt}`).join(', '),
    values: items,
  };
    """
    captured = capture_varnames(program)
    new_program = rename_variable(program, "myvar", captured["items"])
    gold = """
export function buildInClause<T>(
  myvar,
  startingAt = 1,
) {
  return {
    in: myvar.map((_, i) => `$${i + startingAt}`).join(', '),
    values: myvar,
  };
    """
    assert new_program == gold, new_program
    
def test_rename_vars2():
    program = """
export function buildInClause<T>(
  items,
  startingAt = 1,
) {
  return {
    in: items.map((_, i) => `$${i + startingAt}`).join(', '),
    values: items,
  };
    """
    captured = capture_varnames(program)
    newname = make_new_name("items", captured)
    assert newname != "items", newname
    new_program = rename_variable(program, newname, captured["items"])
    gold = program.replace("items", newname)
    assert new_program == gold, new_program
    # tokenizer = AutoTokenizer.from_pretrained("/home/arjun/models/starcoderbase-1b")
    # tokens = tokenizer.tokenize(newname)
    # assert len(tokens) == 1, tokens
    
# TODO: test on real programs by renaming then undoing the rename, checking that the program is the same
def test_on_dataset():
    random.seed(42)
    ds = datasets.load_dataset("franlucc/stenotype-eval-renamed-v4", split="train")
    programs = ds["original_prog"]
    for i,p in enumerate(tqdm(programs[:200])):
        captured = capture_varnames(p)
        random_var = random.choice(list(captured.keys()))
        newname = make_new_name(random_var, captured)
        if newname == None:
            continue
        print(f"Renaming {random_var} to {newname}")
        new_program = rename_variable(p, newname, captured[random_var])
        assert new_program != p, "Same program after renaming"
        new_program = rename_variable(new_program, random_var, captured[random_var])
        if new_program != p:
            with open("renamed_test.ts", "w") as f:
                f.write(new_program)
            with open("orig_test.ts", "w") as f:
                f.write(p)
                
        assert new_program == p, "Program not the same after renaming and undoing"
        
if __name__ == "__main__":
    test_rename_vars()
    test_rename_vars2()
    test_on_dataset()
    print("All tests passed!")
    