from codetrace.type_inf_exp.rename_vars import *
from transformers import AutoTokenizer

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
    new_program = rename_variable(program, "myvar", captured[b"items"])
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
    new_program = rename_variable(program, newname, captured[b"items"])
    gold = program.replace("items", newname)
    assert new_program == gold, new_program
    # tokenizer = AutoTokenizer.from_pretrained("/home/arjun/models/starcoderbase-1b")
    # tokens = tokenizer.tokenize(newname)
    # assert len(tokens) == 1, tokens
    
# TODO: test on real programs by renaming then undoing the rename, checking that the program is the same
    
if __name__ == "__main__":
    test_rename_vars()
    test_rename_vars2()
    print("All tests passed!")
    