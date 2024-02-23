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
    
def test_rename_vars3():
    program = """const t : any = 'a';"""
    captured = capture_varnames(program)
    assert "any" not in captured, captured
    
    program = open("tests/test_prog.ts").read()
    tree = TS_PARSER.parse(bytes( program, "utf8"))
    captures = query.captures(tree.root_node)
    for c in captures:
      if c[0].text.decode("utf-8") == "any":
        print(c[0].text, c)
      elif c[0].text.decode("utf-8") == "typeMap":
        print(c[0].text, c)
    
    var_locs = capture_varnames(program)
    for v, l in var_locs.items():
      if v == "any":
        print(v, l)
      elif v == "typeMap":
        print(v, l)
    assert "typeMap" in var_locs, "typeMap should be captured"
    assert "any" not in var_locs, "Any type should not be captured"


if __name__ == "__main__":
    test_rename_vars()
    test_rename_vars2()
    test_rename_vars3()
    print("All tests passed!")
    