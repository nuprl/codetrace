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
  tree = TS_PARSER.parse(bytes( program, "utf8"))
  captured = capture_varnames(tree)
  new_name = make_new_name("items", set(captured.keys()))
  new_program = rename_variable(tree.text, "myvar", captured["items"]).strip()
  gold = """
export function buildInClause<T>(
myvar,
startingAt = 1,
) {
return {
  in: myvar.map((_, i) => `$${i + startingAt}`).join(', '),
  values: myvar,
};
  """.strip()
  assert new_program == gold, new_program + "\n" + gold
    
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
  tree = TS_PARSER.parse(bytes( program, "utf8"))
  captured = capture_varnames(tree)
  newname = make_new_name("items", set(captured.keys()))
  assert newname != "items", newname
  new_program = rename_variable(tree.text, newname, captured["items"]).strip()
  gold = program.replace("items", newname).strip()
  assert new_program == gold, new_program
  # tokenizer = AutoTokenizer.from_pretrained("/home/arjun/models/starcoderbase-1b")
  # tokens = tokenizer.tokenize(newname)
  # assert len(tokens) == 1, tokens
    
def test_rename_vars3():
  program = """const t : any = 'a';"""
  tree = TS_PARSER.parse(bytes( program, "utf8"))
  captured = capture_varnames(tree)
  assert "any" not in captured, captured
  
  program = open("tests/test_prog.ts").read()
  tree = TS_PARSER.parse(bytes( program, "utf8"))
  captures = query.captures(tree.root_node)
  for c in captures:
    if c[0].text.decode("utf-8") == "any":
      print(c[0].text, c)
    elif c[0].text.decode("utf-8") == "typeMap":
      print(c[0].text, c)
  
  tree = TS_PARSER.parse(bytes( program, "utf8"))
  var_locs = capture_varnames(tree)
  for v, l in var_locs.items():
    if v == "any":
      print(v, l)
    elif v == "typeMap":
      print(v, l)
  assert "typeMap" in var_locs, "typeMap should be captured"
  assert "any" not in var_locs, "Any type should not be captured"

def test_remove_comments():
  prog = """// These functions will throw an error if the JSON doesn't
// match the expected interface, even if the JSON is valid.
/* weee
*
*/
export interface GrypeCvss {
    VendorMetadata: any; //test
    Metrics: Metrics;
    Vector: string;
    Version: string;
  } // hi
  """
  gold = """export interface GrypeCvss {
    VendorMetadata: any; 
    Metrics: Metrics;
    Vector: string;
    Version: string;
  }"""
  assert remove_comments(prog) == gold, remove_comments(prog)

if __name__ == "__main__":
    test_rename_vars()
    test_rename_vars2()
    # test_rename_vars3()
    test_remove_comments()
    print("All tests passed!")
    