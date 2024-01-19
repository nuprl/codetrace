declare var require: any;
const assert = require("node:assert");
type cmp = { kind: "EQ" } | { kind: "GT" } | { kind: "LT" };
type nat = { kind: "Z" } | { kind: "S"; f0: nat };
type tree = { kind: "Leaf" } | { kind: "Node"; f0: tree; f1: nat; f2: tree };

function comp_nat(n1: nat, n2: nat): cmp {
  switch (n1.kind) {
    case "Z": {
      switch (n2.kind) {
        case "Z": {
          return { kind: "EQ" };
        }
        case "S": {
          let n2p = n2.f0;
          return { kind: "LT" };
        }
      }
    }
    case "S": {
      let n3 = n1.f0;
      switch (n2.kind) {
        case "Z": {
          return { kind: "GT" };
        }
        case "S": {
          let n4 = n2.f0;
          return comp_nat(n3, n4);
        }
      }
    }
  }
}

function tree_binsert(t: tree, n: nat): tree {
  switch (t.kind) {
    case "Leaf": {
      return {
        kind: "Node",
        f0: { kind: "Leaf" },
        f1: n,
        f2: { kind: "Leaf" },
      };
    }
    case "Node": {
      let r = t.f2;
      let d = t.f1;
      let l = t.f0;
      switch (comp_nat(n, d).kind) {
        case "EQ": {
          return { kind: "Node", f0: l, f1: d, f2: r };
        }
        case "LT": {
          return { kind: "Node", f0: tree_binsert(l, n), f1: d, f2: r };
        }
        case "GT": {
          return { kind: "Node", f0: l, f1: d, f2: tree_binsert(r, n) };
        }
      }
    }
  }
}

function assertions() {
  assert.deepEqual(tree_binsert({ kind: "Leaf" }, { kind: "Z" }), {
    kind: "Node",
    f0: { kind: "Leaf" },
    f1: { kind: "Z" },
    f2: { kind: "Leaf" },
  });
  assert.deepEqual(
    tree_binsert({ kind: "Leaf" }, { kind: "S", f0: { kind: "Z" } }),
    {
      kind: "Node",
      f0: { kind: "Leaf" },
      f1: { kind: "S", f0: { kind: "Z" } },
      f2: { kind: "Leaf" },
    }
  );
  assert.deepEqual(
    tree_binsert(
      { kind: "Leaf" },
      { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } }
    ),
    {
      kind: "Node",
      f0: { kind: "Leaf" },
      f1: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
      f2: { kind: "Leaf" },
    }
  );
  assert.deepEqual(
    tree_binsert(
      {
        kind: "Node",
        f0: { kind: "Leaf" },
        f1: { kind: "Z" },
        f2: { kind: "Leaf" },
      },
      { kind: "Z" }
    ),
    {
      kind: "Node",
      f0: { kind: "Leaf" },
      f1: { kind: "Z" },
      f2: { kind: "Leaf" },
    }
  );
  assert.deepEqual(
    tree_binsert(
      {
        kind: "Node",
        f0: { kind: "Leaf" },
        f1: { kind: "Z" },
        f2: { kind: "Leaf" },
      },
      { kind: "S", f0: { kind: "Z" } }
    ),
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
    tree_binsert(
      {
        kind: "Node",
        f0: { kind: "Leaf" },
        f1: { kind: "Z" },
        f2: { kind: "Leaf" },
      },
      { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } }
    ),
    {
      kind: "Node",
      f0: { kind: "Leaf" },
      f1: { kind: "Z" },
      f2: {
        kind: "Node",
        f0: { kind: "Leaf" },
        f1: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
        f2: { kind: "Leaf" },
      },
    }
  );
  assert.deepEqual(
    tree_binsert(
      {
        kind: "Node",
        f0: { kind: "Leaf" },
        f1: { kind: "S", f0: { kind: "Z" } },
        f2: { kind: "Leaf" },
      },
      { kind: "Z" }
    ),
    {
      kind: "Node",
      f0: {
        kind: "Node",
        f0: { kind: "Leaf" },
        f1: { kind: "Z" },
        f2: { kind: "Leaf" },
      },
      f1: { kind: "S", f0: { kind: "Z" } },
      f2: { kind: "Leaf" },
    }
  );
  assert.deepEqual(
    tree_binsert(
      {
        kind: "Node",
        f0: { kind: "Leaf" },
        f1: { kind: "S", f0: { kind: "Z" } },
        f2: { kind: "Leaf" },
      },
      { kind: "S", f0: { kind: "Z" } }
    ),
    {
      kind: "Node",
      f0: { kind: "Leaf" },
      f1: { kind: "S", f0: { kind: "Z" } },
      f2: { kind: "Leaf" },
    }
  );
  assert.deepEqual(
    tree_binsert(
      {
        kind: "Node",
        f0: { kind: "Leaf" },
        f1: { kind: "S", f0: { kind: "Z" } },
        f2: { kind: "Leaf" },
      },
      { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } }
    ),
    {
      kind: "Node",
      f0: { kind: "Leaf" },
      f1: { kind: "S", f0: { kind: "Z" } },
      f2: {
        kind: "Node",
        f0: { kind: "Leaf" },
        f1: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
        f2: { kind: "Leaf" },
      },
    }
  );
  assert.deepEqual(
    tree_binsert(
      {
        kind: "Node",
        f0: { kind: "Leaf" },
        f1: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
        f2: { kind: "Leaf" },
      },
      { kind: "Z" }
    ),
    {
      kind: "Node",
      f0: {
        kind: "Node",
        f0: { kind: "Leaf" },
        f1: { kind: "Z" },
        f2: { kind: "Leaf" },
      },
      f1: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
      f2: { kind: "Leaf" },
    }
  );
  assert.deepEqual(
    tree_binsert(
      {
        kind: "Node",
        f0: { kind: "Leaf" },
        f1: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
        f2: { kind: "Leaf" },
      },
      { kind: "S", f0: { kind: "Z" } }
    ),
    {
      kind: "Node",
      f0: {
        kind: "Node",
        f0: { kind: "Leaf" },
        f1: { kind: "S", f0: { kind: "Z" } },
        f2: { kind: "Leaf" },
      },
      f1: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
      f2: { kind: "Leaf" },
    }
  );
  assert.deepEqual(
    tree_binsert(
      {
        kind: "Node",
        f0: { kind: "Leaf" },
        f1: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
        f2: { kind: "Leaf" },
      },
      { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } }
    ),
    {
      kind: "Node",
      f0: { kind: "Leaf" },
      f1: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
      f2: { kind: "Leaf" },
    }
  );
  assert.deepEqual(
    tree_binsert(
      {
        kind: "Node",
        f0: {
          kind: "Node",
          f0: { kind: "Leaf" },
          f1: { kind: "Z" },
          f2: { kind: "Leaf" },
        },
        f1: { kind: "S", f0: { kind: "Z" } },
        f2: { kind: "Leaf" },
      },
      { kind: "Z" }
    ),
    {
      kind: "Node",
      f0: {
        kind: "Node",
        f0: { kind: "Leaf" },
        f1: { kind: "Z" },
        f2: { kind: "Leaf" },
      },
      f1: { kind: "S", f0: { kind: "Z" } },
      f2: { kind: "Leaf" },
    }
  );
  assert.deepEqual(
    tree_binsert(
      {
        kind: "Node",
        f0: {
          kind: "Node",
          f0: { kind: "Leaf" },
          f1: { kind: "Z" },
          f2: { kind: "Leaf" },
        },
        f1: { kind: "S", f0: { kind: "Z" } },
        f2: { kind: "Leaf" },
      },
      { kind: "S", f0: { kind: "Z" } }
    ),
    {
      kind: "Node",
      f0: {
        kind: "Node",
        f0: { kind: "Leaf" },
        f1: { kind: "Z" },
        f2: { kind: "Leaf" },
      },
      f1: { kind: "S", f0: { kind: "Z" } },
      f2: { kind: "Leaf" },
    }
  );
  assert.deepEqual(
    tree_binsert(
      {
        kind: "Node",
        f0: {
          kind: "Node",
          f0: { kind: "Leaf" },
          f1: { kind: "Z" },
          f2: { kind: "Leaf" },
        },
        f1: { kind: "S", f0: { kind: "Z" } },
        f2: { kind: "Leaf" },
      },
      { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } }
    ),
    {
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
        f1: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
        f2: { kind: "Leaf" },
      },
    }
  );
  assert.deepEqual(
    tree_binsert(
      {
        kind: "Node",
        f0: { kind: "Leaf" },
        f1: { kind: "S", f0: { kind: "Z" } },
        f2: {
          kind: "Node",
          f0: { kind: "Leaf" },
          f1: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
          f2: { kind: "Leaf" },
        },
      },
      { kind: "Z" }
    ),
    {
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
        f1: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
        f2: { kind: "Leaf" },
      },
    }
  );
  assert.deepEqual(
    tree_binsert(
      {
        kind: "Node",
        f0: { kind: "Leaf" },
        f1: { kind: "S", f0: { kind: "Z" } },
        f2: {
          kind: "Node",
          f0: { kind: "Leaf" },
          f1: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
          f2: { kind: "Leaf" },
        },
      },
      { kind: "S", f0: { kind: "Z" } }
    ),
    {
      kind: "Node",
      f0: { kind: "Leaf" },
      f1: { kind: "S", f0: { kind: "Z" } },
      f2: {
        kind: "Node",
        f0: { kind: "Leaf" },
        f1: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
        f2: { kind: "Leaf" },
      },
    }
  );
}
assertions();

function validations() {
  assert.deepEqual(
    tree_binsert(
      {
        kind: "Node",
        f0: { kind: "Leaf" },
        f1: { kind: "Z" },
        f2: {
          kind: "Node",
          f0: { kind: "Leaf" },
          f1: { kind: "S", f0: { kind: "Z" } },
          f2: {
            kind: "Node",
            f0: { kind: "Leaf" },
            f1: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
            f2: { kind: "Leaf" },
          },
        },
      },
      { kind: "S", f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } } }
    ),
    {
      kind: "Node",
      f0: { kind: "Leaf" },
      f1: { kind: "Z" },
      f2: {
        kind: "Node",
        f0: { kind: "Leaf" },
        f1: { kind: "S", f0: { kind: "Z" } },
        f2: {
          kind: "Node",
          f0: { kind: "Leaf" },
          f1: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
          f2: {
            kind: "Node",
            f0: { kind: "Leaf" },
            f1: {
              kind: "S",
              f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
            },
            f2: { kind: "Leaf" },
          },
        },
      },
    }
  );
  assert.deepEqual(
    tree_binsert(
      {
        kind: "Node",
        f0: {
          kind: "Node",
          f0: {
            kind: "Node",
            f0: { kind: "Leaf" },
            f1: { kind: "S", f0: { kind: "Z" } },
            f2: { kind: "Leaf" },
          },
          f1: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
          f2: { kind: "Leaf" },
        },
        f1: {
          kind: "S",
          f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
        },
        f2: { kind: "Leaf" },
      },
      { kind: "Z" }
    ),
    {
      kind: "Node",
      f0: {
        kind: "Node",
        f0: {
          kind: "Node",
          f0: {
            kind: "Node",
            f0: { kind: "Leaf" },
            f1: { kind: "Z" },
            f2: { kind: "Leaf" },
          },
          f1: { kind: "S", f0: { kind: "Z" } },
          f2: { kind: "Leaf" },
        },
        f1: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
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
    tree_binsert(
      {
        kind: "Node",
        f0: {
          kind: "Node",
          f0: { kind: "Leaf" },
          f1: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
          f2: { kind: "Leaf" },
        },
        f1: {
          kind: "S",
          f0: {
            kind: "S",
            f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
          },
        },
        f2: {
          kind: "Node",
          f0: { kind: "Leaf" },
          f1: {
            kind: "S",
            f0: {
              kind: "S",
              f0: {
                kind: "S",
                f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
              },
            },
          },
          f2: { kind: "Leaf" },
        },
      },
      { kind: "S", f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } } }
    ),
    {
      kind: "Node",
      f0: {
        kind: "Node",
        f0: { kind: "Leaf" },
        f1: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
        f2: {
          kind: "Node",
          f0: { kind: "Leaf" },
          f1: {
            kind: "S",
            f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
          },
          f2: { kind: "Leaf" },
        },
      },
      f1: {
        kind: "S",
        f0: {
          kind: "S",
          f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
        },
      },
      f2: {
        kind: "Node",
        f0: { kind: "Leaf" },
        f1: {
          kind: "S",
          f0: {
            kind: "S",
            f0: {
              kind: "S",
              f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
            },
          },
        },
        f2: { kind: "Leaf" },
      },
    }
  );
  assert.deepEqual(
    tree_binsert(
      {
        kind: "Node",
        f0: { kind: "Leaf" },
        f1: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
        f2: {
          kind: "Node",
          f0: { kind: "Leaf" },
          f1: {
            kind: "S",
            f0: {
              kind: "S",
              f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
            },
          },
          f2: { kind: "Leaf" },
        },
      },
      { kind: "S", f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } } }
    ),
    {
      kind: "Node",
      f0: { kind: "Leaf" },
      f1: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
      f2: {
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
          f0: {
            kind: "S",
            f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
          },
        },
        f2: { kind: "Leaf" },
      },
    }
  );
}
validations();
