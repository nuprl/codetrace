declare var require: any;
const assert = require("node:assert");
type nat = { kind: "Z" } | { kind: "S"; f0: nat };
type tree = { kind: "Leaf" } | { kind: "Node"; f0: nat; f1: tree; f2: tree };
type bbool = { kind: "True" } | { kind: "False" };
type cmp = { kind: "LT" } | { kind: "EQ" } | { kind: "GT" };

function compare(x1: nat, x2: nat): cmp {
  switch (x1.kind) {
    case "Z": {
      switch (x2.kind) {
        case "Z": {
          return { kind: "EQ" };
        }
        case "S": {
          let n1 = x2.f0;
          return { kind: "LT" };
        }
      }
    }
    case "S": {
      let n1 = x1.f0;
      switch (x2.kind) {
        case "Z": {
          return { kind: "GT" };
        }
        case "S": {
          let n2 = x2.f0;
          return compare(n1, n2);
        }
      }
    }
  }
}
function max(n1: nat, n2: nat): nat {
  switch (compare(n1, n2).kind) {
    case "LT": {
      return n2;
    }
    case "EQ": {
      return n1;
    }
    case "GT": {
      return n1;
    }
  }
}
function height(t: tree): nat {
  switch (t.kind) {
    case "Leaf": {
      return { kind: "Z" };
    }
    case "Node": {
      let rt = t.f2;
      let lt = t.f1;
      let v = t.f0;
      switch (lt.kind) {
        case "Leaf": {
          return { kind: "S", f0: height(rt) };
        }
        case "Node": {
          let lrt = lt.f2;
          let llt = lt.f1;
          let lv = lt.f0;
          switch (rt.kind) {
            case "Leaf": {
              return { kind: "S", f0: height(lt) };
            }
            case "Node": {
              let rrt = rt.f2;
              let rlt = rt.f1;
              let rv = rt.f0;
              return { kind: "S", f0: max(height(lt), height(rt)) };
            }
          }
        }
      }
    }
  }
}
function bool_band(b1: bbool, b2: bbool): bbool {
  switch (b1.kind) {
    case "True": {
      return b2;
    }
    case "False": {
      return { kind: "False" };
    }
  }
}

function tree_balanced(t: tree): bbool {
  switch (t.kind) {
    case "Leaf": {
      return { kind: "True" };
    }
    case "Node": {
      let rt = t.f2;
      let lt = t.f1;
      let v = t.f0;
      let lth: nat = height(lt);
      let rth: nat = height(rt);
      switch (compare(lth, rth).kind) {
        case "EQ": {
          return bool_band(tree_balanced(lt), tree_balanced(rt));
        }
        case "LT": {
          switch (rth.kind) {
            case "Z": {
              return { kind: "False" };
            }
            case "S": {
              let n = rth.f0;
              switch (compare(lth, n).kind) {
                case "EQ": {
                  return bool_band(tree_balanced(lt), tree_balanced(rt));
                }
                case "LT": {
                  return { kind: "False" };
                }
                case "GT": {
                  return { kind: "False" };
                }
              }
            }
          }
        }
        case "GT": {
          switch (lth.kind) {
            case "Z": {
              return { kind: "False" };
            }
            case "S": {
              let n = lth.f0;
              switch (compare(rth, n).kind) {
                case "EQ": {
                  return bool_band(tree_balanced(lt), tree_balanced(rt));
                }
                case "LT": {
                  return { kind: "False" };
                }
                case "GT": {
                  return { kind: "False" };
                }
              }
            }
          }
        }
      }
    }
  }
}

function assertions() {
  assert.deepEqual(tree_balanced({ kind: "Leaf" }), { kind: "True" });
  assert.deepEqual(
    tree_balanced({
      kind: "Node",
      f0: { kind: "Z" },
      f1: { kind: "Leaf" },
      f2: { kind: "Leaf" },
    }),
    { kind: "True" }
  );
  assert.deepEqual(
    tree_balanced({
      kind: "Node",
      f0: { kind: "Z" },
      f1: {
        kind: "Node",
        f0: { kind: "S", f0: { kind: "Z" } },
        f1: { kind: "Leaf" },
        f2: { kind: "Leaf" },
      },
      f2: { kind: "Leaf" },
    }),
    { kind: "True" }
  );
  assert.deepEqual(
    tree_balanced({
      kind: "Node",
      f0: { kind: "Z" },
      f1: {
        kind: "Node",
        f0: { kind: "S", f0: { kind: "Z" } },
        f1: {
          kind: "Node",
          f0: { kind: "S", f0: { kind: "Z" } },
          f1: { kind: "Leaf" },
          f2: { kind: "Leaf" },
        },
        f2: { kind: "Leaf" },
      },
      f2: { kind: "Leaf" },
    }),
    { kind: "False" }
  );
  assert.deepEqual(
    tree_balanced({
      kind: "Node",
      f0: { kind: "Z" },
      f1: {
        kind: "Node",
        f0: { kind: "S", f0: { kind: "Z" } },
        f1: {
          kind: "Node",
          f0: { kind: "S", f0: { kind: "Z" } },
          f1: { kind: "Leaf" },
          f2: { kind: "Leaf" },
        },
        f2: {
          kind: "Node",
          f0: { kind: "S", f0: { kind: "Z" } },
          f1: { kind: "Leaf" },
          f2: { kind: "Leaf" },
        },
      },
      f2: {
        kind: "Node",
        f0: { kind: "S", f0: { kind: "Z" } },
        f1: { kind: "Leaf" },
        f2: { kind: "Leaf" },
      },
    }),
    { kind: "True" }
  );
  assert.deepEqual(
    tree_balanced({
      kind: "Node",
      f0: { kind: "Z" },
      f1: {
        kind: "Node",
        f0: { kind: "S", f0: { kind: "Z" } },
        f1: {
          kind: "Node",
          f0: { kind: "S", f0: { kind: "Z" } },
          f1: { kind: "Leaf" },
          f2: { kind: "Leaf" },
        },
        f2: {
          kind: "Node",
          f0: { kind: "S", f0: { kind: "Z" } },
          f1: { kind: "Leaf" },
          f2: {
            kind: "Node",
            f0: { kind: "S", f0: { kind: "Z" } },
            f1: { kind: "Leaf" },
            f2: { kind: "Leaf" },
          },
        },
      },
      f2: {
        kind: "Node",
        f0: { kind: "S", f0: { kind: "Z" } },
        f1: { kind: "Leaf" },
        f2: { kind: "Leaf" },
      },
    }),
    { kind: "False" }
  );
}
assertions();

function validations() {
  assert.deepEqual(
    tree_balanced({
      kind: "Node",
      f0: { kind: "Z" },
      f1: {
        kind: "Node",
        f0: { kind: "S", f0: { kind: "Z" } },
        f1: {
          kind: "Node",
          f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
          f1: { kind: "Leaf" },
          f2: { kind: "Leaf" },
        },
        f2: { kind: "Leaf" },
      },
      f2: { kind: "Leaf" },
    }),
    { kind: "False" }
  );
  assert.deepEqual(
    tree_balanced({
      kind: "Node",
      f0: { kind: "Z" },
      f1: { kind: "Leaf" },
      f2: {
        kind: "Node",
        f0: { kind: "S", f0: { kind: "Z" } },
        f1: {
          kind: "Node",
          f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
          f1: { kind: "Leaf" },
          f2: { kind: "Leaf" },
        },
        f2: { kind: "Leaf" },
      },
    }),
    { kind: "False" }
  );
  assert.deepEqual(
    tree_balanced({
      kind: "Node",
      f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
      f1: {
        kind: "Node",
        f0: {
          kind: "S",
          f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
        },
        f1: { kind: "Leaf" },
        f2: { kind: "Leaf" },
      },
      f2: {
        kind: "Node",
        f0: { kind: "Z" },
        f1: { kind: "Leaf" },
        f2: { kind: "Leaf" },
      },
    }),
    { kind: "True" }
  );
  assert.deepEqual(
    tree_balanced({
      kind: "Node",
      f0: {
        kind: "S",
        f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
      },
      f1: {
        kind: "Node",
        f0: { kind: "Z" },
        f1: {
          kind: "Node",
          f0: { kind: "S", f0: { kind: "Z" } },
          f1: { kind: "Leaf" },
          f2: { kind: "Leaf" },
        },
        f2: {
          kind: "Node",
          f0: {
            kind: "S",
            f0: {
              kind: "S",
              f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
            },
          },
          f1: { kind: "Leaf" },
          f2: { kind: "Leaf" },
        },
      },
      f2: {
        kind: "Node",
        f0: { kind: "S", f0: { kind: "Z" } },
        f1: { kind: "Leaf" },
        f2: { kind: "Leaf" },
      },
    }),
    { kind: "True" }
  );
}
validations();
