declare var require: any;
const assert = require("node:assert");
type nat = { kind: "Z" } | { kind: "S"; f0: nat };
type bbool = { kind: "True" } | { kind: "False" };
type tree = { kind: "Leaf" } | { kind: "Node"; f0: nat; f1: tree; f2: tree };

function tree_lastleft(t: tree): nat {
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
          return v;
        }
        case "Node": {
          let lrt = lt.f2;
          let llt = lt.f1;
          let lv = lt.f0;
          return tree_lastleft(lt);
        }
      }
    }
  }
}

function assertions() {
  assert.deepEqual(tree_lastleft({ kind: "Leaf" }), { kind: "Z" });
  assert.deepEqual(
    tree_lastleft({
      kind: "Node",
      f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
      f1: { kind: "Leaf" },
      f2: { kind: "Leaf" },
    }),
    { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } }
  );
  assert.deepEqual(
    tree_lastleft({
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
    { kind: "S", f0: { kind: "Z" } }
  );
  assert.deepEqual(
    tree_lastleft({
      kind: "Node",
      f0: { kind: "Z" },
      f1: {
        kind: "Node",
        f0: {
          kind: "S",
          f0: {
            kind: "S",
            f0: {
              kind: "S",
              f0: {
                kind: "S",
                f0: {
                  kind: "S",
                  f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
                },
              },
            },
          },
        },
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
    { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } }
  );
  assert.deepEqual(
    tree_lastleft({
      kind: "Node",
      f0: { kind: "Z" },
      f1: {
        kind: "Node",
        f0: { kind: "S", f0: { kind: "Z" } },
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
        f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
        f1: { kind: "Leaf" },
        f2: { kind: "Leaf" },
      },
    }),
    { kind: "S", f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } } }
  );
  assert.deepEqual(
    tree_lastleft({
      kind: "Node",
      f0: { kind: "Z" },
      f1: {
        kind: "Node",
        f0: { kind: "S", f0: { kind: "Z" } },
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
          f0: {
            kind: "S",
            f0: {
              kind: "S",
              f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
            },
          },
          f1: { kind: "Leaf" },
          f2: {
            kind: "Node",
            f0: {
              kind: "S",
              f0: {
                kind: "S",
                f0: {
                  kind: "S",
                  f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
                },
              },
            },
            f1: { kind: "Leaf" },
            f2: { kind: "Leaf" },
          },
        },
      },
      f2: {
        kind: "Node",
        f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
        f1: { kind: "Leaf" },
        f2: { kind: "Leaf" },
      },
    }),
    { kind: "S", f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } } }
  );
}
assertions();

function validations() {
  assert.deepEqual(
    tree_lastleft({
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
    { kind: "Z" }
  );
  assert.deepEqual(
    tree_lastleft({
      kind: "Node",
      f0: {
        kind: "S",
        f0: {
          kind: "S",
          f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
        },
      },
      f1: { kind: "Leaf" },
      f2: {
        kind: "Node",
        f0: { kind: "Z" },
        f1: { kind: "Leaf" },
        f2: {
          kind: "Node",
          f0: { kind: "Z" },
          f1: { kind: "Leaf" },
          f2: { kind: "Leaf" },
        },
      },
    }),
    {
      kind: "S",
      f0: {
        kind: "S",
        f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
      },
    }
  );
  assert.deepEqual(
    tree_lastleft({
      kind: "Node",
      f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
      f1: {
        kind: "Node",
        f0: {
          kind: "S",
          f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
        },
        f1: {
          kind: "Node",
          f0: { kind: "Z" },
          f1: { kind: "Leaf" },
          f2: { kind: "Leaf" },
        },
        f2: {
          kind: "Node",
          f0: { kind: "Z" },
          f1: { kind: "Leaf" },
          f2: { kind: "Leaf" },
        },
      },
      f2: { kind: "Leaf" },
    }),
    { kind: "Z" }
  );
}
validations();
