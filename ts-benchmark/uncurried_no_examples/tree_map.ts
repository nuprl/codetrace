declare var require: any;
const assert = require("node:assert");
type nat = { kind: "Z" } | { kind: "S"; f0: nat };
type nlist = { kind: "Nil" } | { kind: "Cons"; f0: nat; f1: nlist };
type tree = { kind: "Leaf" } | { kind: "Node"; f0: tree; f1: nat; f2: tree };

function div2(n: nat): nat {
  switch (n.kind) {
    case "Z": {
      return { kind: "Z" };
    }
    case "S": {
      let n1 = n.f0;
      switch (n1.kind) {
        case "Z": {
          return { kind: "Z" };
        }
        case "S": {
          let n2 = n1.f0;
          return { kind: "S", f0: div2(n2) };
        }
      }
    }
  }
}
function inc(n: nat): nat {
  return { kind: "S", f0: n };
}

function tree_map(f: (__x9: nat) => nat, t: tree): tree {
  switch (t.kind) {
    case "Leaf": {
      return { kind: "Leaf" };
    }
    case "Node": {
      let r = t.f2;
      let n = t.f1;
      let l = t.f0;
      return { kind: "Node", f0: tree_map(f, l), f1: f(n), f2: tree_map(f, r) };
    }
  }
}

function assertions() {
  assert.deepEqual(tree_map(div2, { kind: "Leaf" }), { kind: "Leaf" });
  assert.deepEqual(
    tree_map(div2, {
      kind: "Node",
      f0: {
        kind: "Node",
        f0: { kind: "Leaf" },
        f1: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
        f2: { kind: "Leaf" },
      },
      f1: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
      f2: { kind: "Leaf" },
    }),
    {
      kind: "Node",
      f0: {
        kind: "Node",
        f0: { kind: "Leaf" },
        f1: { kind: "S", f0: { kind: "Z" } },
        f2: { kind: "Leaf" },
      },
      f1: { kind: "S", f0: { kind: "Z" } },
      f2: { kind: "Leaf" },
    }
  );
  assert.deepEqual(
    tree_map(div2, {
      kind: "Node",
      f0: { kind: "Leaf" },
      f1: { kind: "S", f0: { kind: "Z" } },
      f2: {
        kind: "Node",
        f0: { kind: "Leaf" },
        f1: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
        f2: { kind: "Leaf" },
      },
    }),
    {
      kind: "Node",
      f0: { kind: "Leaf" },
      f1: { kind: "Z" },
      f2: {
        kind: "Node",
        f0: { kind: "Leaf" },
        f1: { kind: "S", f0: { kind: "Z" } },
        f2: { kind: "Leaf" },
      },
    }
  );
  assert.deepEqual(
    tree_map(inc, {
      kind: "Node",
      f0: { kind: "Leaf" },
      f1: { kind: "Z" },
      f2: { kind: "Leaf" },
    }),
    {
      kind: "Node",
      f0: { kind: "Leaf" },
      f1: { kind: "S", f0: { kind: "Z" } },
      f2: { kind: "Leaf" },
    }
  );
}
assertions();

function validations() {
  assert.deepEqual(
    tree_map(inc, {
      kind: "Node",
      f0: {
        kind: "Node",
        f0: { kind: "Leaf" },
        f1: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
        f2: { kind: "Leaf" },
      },
      f1: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
      f2: { kind: "Leaf" },
    }),
    {
      kind: "Node",
      f0: {
        kind: "Node",
        f0: { kind: "Leaf" },
        f1: {
          kind: "S",
          f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
        },
        f2: { kind: "Leaf" },
      },
      f1: {
        kind: "S",
        f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
      },
      f2: { kind: "Leaf" },
    }
  );
  assert.deepEqual(
    tree_map(
      function (n: nat) {
        switch (n.kind) {
          case "Z": {
            return { kind: "Z" };
          }
          case "S": {
            let np = n.f0;
            return np;
          }
        }
      },
      {
        kind: "Node",
        f0: { kind: "Leaf" },
        f1: { kind: "S", f0: { kind: "Z" } },
        f2: {
          kind: "Node",
          f0: {
            kind: "Node",
            f0: { kind: "Leaf" },
            f1: { kind: "Z" },
            f2: { kind: "Leaf" },
          },
          f1: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
          f2: {
            kind: "Node",
            f0: { kind: "Leaf" },
            f1: { kind: "S", f0: { kind: "Z" } },
            f2: { kind: "Leaf" },
          },
        },
      }
    ),
    {
      kind: "Node",
      f0: { kind: "Leaf" },
      f1: { kind: "Z" },
      f2: {
        kind: "Node",
        f0: {
          kind: "Node",
          f0: { kind: "Leaf" },
          f1: { kind: "Z" },
          f2: { kind: "Leaf" },
        },
        f1: { kind: "S", f0: { kind: "Z" } },
        f2: {
          kind: "Node",
          f0: { kind: "Leaf" },
          f1: { kind: "Z" },
          f2: { kind: "Leaf" },
        },
      },
    }
  );
  assert.deepEqual(
    tree_map(
      function (n: nat) {
        return n;
      },
      {
        kind: "Node",
        f0: { kind: "Leaf" },
        f1: {
          kind: "S",
          f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
        },
        f2: { kind: "Leaf" },
      }
    ),
    {
      kind: "Node",
      f0: { kind: "Leaf" },
      f1: {
        kind: "S",
        f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
      },
      f2: { kind: "Leaf" },
    }
  );
}
validations();
