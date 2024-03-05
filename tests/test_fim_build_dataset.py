from codetrace.type_inf_exp.build_dataset import *

def test_capture_func_types():
  prog= """export class ColorSet {
  prmColor;
  secColor;
  secBrdColor;
  thrColor;
  thrBrdColor;
  fontColor;
  borderColor;
  arr_baseIntensity;

  constructor(znqatn, color2?, color3?) {
    this.arr_baseIntensity = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "d", "e", "f"];
    this.prmColor = znqatn;
    this.secColor = (color2) ? color2 : this.algo0(znqatn, 15);
    this.secBrdColor = this.algo0(this.secColor, 20);
    this.thrColor = (color3) ? color3 : this.algo0(this.secColor, 25);
    this.thrBrdColor = this.algo0(this.thrColor, 50);
    this.fontColor = this.algo0(this.thrColor, 80);
  }

  algo0(hexa_color: string, diff: number) {
    return this.rgbToHexa(this.subtractMultiply(hexa_color, diff));
  }"""
  tree = TS_PARSER.parse(bytes(prog, "utf-8"))
  query = TS_LANGUAGE.query(TS_QUERY_FUNC_TYPES)
  captures = query.captures(tree.root_node)
  assert len(captures) == 2, captures
  text_cap = [c[0].text for c in captures]
  
def test_remove_types():
    ts_prog = "function foo(a: number, b: string): number {\n\treturn 1;\n}"
    func_fim_prompts = fim_remove_types(ts_prog, TS_QUERY_FUNC_TYPES)
    
    gold = [("function foo(a, b): <FILL> {\n\treturn 1;\n}", "number"),
                  ("function foo(a: <FILL>, b) {\n\treturn 1;\n}", "number"),
                  ("function foo(a, b: <FILL>) {\n\treturn 1;\n}", "string")]
    
    assert set(func_fim_prompts) == set(gold), f"===GOLD===:\n {gold}\n===GOT===:\n{func_fim_prompts}"
    
def test_remove_types_multiline():
  ts_prog = """
class Point {
x: number;
y: number;
}

function foo(a: number, b: string): number | string {
if (a > 0) {
      return 1;
  } else {
      return b;
  }
}
function foo(a: number, b: string): number {
  return 1;
}
"""
  func_fim_prompts = fim_remove_types(ts_prog, TS_QUERY_FUNC_TYPES)
  fim_types = [t for p, t in func_fim_prompts]
  assert fim_types == ["number", "string", "number", "number | string", "string", "number"], fim_types
  reconstruct = [p.replace("<FILL>", t) for p, t in func_fim_prompts]
  assert [r == ts_prog for r in reconstruct], f"===GOLD===:\n {ts_prog}\n===GOT===:\n{reconstruct}"
    
    
def test_remove_types_full():
  ts_prog = """
class Point {
  x: number;
  y: number;
}
 
function foo(a: number, b: string): number | string {
	if (a > 0) {
        return 1;
    } else {
        return b;
    }
}
function greeter(fn: (a: string) => void) {
  fn("Hello, World");
}
"""
  func_fim_prompts = fim_remove_types(ts_prog, TS_QUERY_FUNC_TYPES)
  fim_types = [t for p, t in func_fim_prompts]
  assert fim_types == ["(a: string) => void", "number | string", "string", "number"], fim_types
  reconstruct = [p.replace("<FILL>", t) for p, t in func_fim_prompts]
  assert [r == ts_prog for r in reconstruct], f"===GOLD===:\n {ts_prog}\n===GOT===:\n{reconstruct}"

    
if __name__ == "__main__":
  test_remove_types()
  test_remove_types_multiline()
  test_remove_types_full()
  test_capture_func_types()
  print("All tests passed!")