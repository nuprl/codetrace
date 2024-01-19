declare var require: any;
const assert = require("node:assert");
type nat = { kind: "Z" } | { kind: "S"; f0: nat };
type tree = { kind: "Leaf" } | { kind: "Node"; f0: tree; f1: nat; f2: tree };

function sum(n1: nat): (n2: nat) => nat {
  return function (n2: nat) {
    switch (n1.kind) {
      case "Z": {
        return n2;
      }
      case "S": {
        let n3 = n1.f0;
        return { kind: "S", f0: sum(n3)(n2) };
      }
    }
  };
}

function tree_count_nodes(t: tree): nat {
  switch (t.kind) {
    case "Leaf": {
      return { kind: "Z" };
    }
    case "Node": {
      let r = t.f2;
      let n = t.f1;
      let l = t.f0;
      return { kind: "S", f0: sum(tree_count_nodes(l))(tree_count_nodes(r)) };
    }
  }
}

function assertions() {
  assert.deepEqual(tree_count_nodes({ kind: "Leaf" }), { kind: "Z" });
  assert.deepEqual(
    tree_count_nodes({
      kind: "Node",
      f0: { kind: "Leaf" },
      f1: { kind: "Z" },
      f2: { kind: "Leaf" },
    }),
    { kind: "S", f0: { kind: "Z" } }
  );
  assert.deepEqual(
    tree_count_nodes({
      kind: "Node",
      f0: {
        kind: "Node",
        f0: { kind: "Leaf" },
        f1: { kind: "Z" },
        f2: { kind: "Leaf" },
      },
      f1: { kind: "Z" },
      f2: { kind: "Leaf" },
    }),
    { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } }
  );
  assert.deepEqual(
    tree_count_nodes({
      kind: "Node",
      f0: { kind: "Leaf" },
      f1: { kind: "Z" },
      f2: {
        kind: "Node",
        f0: { kind: "Leaf" },
        f1: { kind: "Z" },
        f2: { kind: "Leaf" },
      },
    }),
    { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } }
  );
  assert.deepEqual(
    tree_count_nodes({
      kind: "Node",
      f0: {
        kind: "Node",
        f0: { kind: "Leaf" },
        f1: { kind: "Z" },
        f2: {
          kind: "Node",
          f0: { kind: "Leaf" },
          f1: { kind: "Z" },
          f2: { kind: "Leaf" },
        },
      },
      f1: { kind: "Z" },
      f2: { kind: "Leaf" },
    }),
    { kind: "S", f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } } }
  );
  assert.deepEqual(
    tree_count_nodes({
      kind: "Node",
      f0: { kind: "Leaf" },
      f1: { kind: "Z" },
      f2: {
        kind: "Node",
        f0: { kind: "Leaf" },
        f1: { kind: "Z" },
        f2: {
          kind: "Node",
          f0: { kind: "Leaf" },
          f1: { kind: "Z" },
          f2: { kind: "Leaf" },
        },
      },
    }),
    { kind: "S", f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } } }
  );
}
assertions();

function validations() {
  assert.deepEqual(
    tree_count_nodes({
      kind: "Node",
      f0: { kind: "Leaf" },
      f1: { kind: "S", f0: { kind: "Z" } },
      f2: {
        kind: "Node",
        f0: {
          kind: "Node",
          f0: { kind: "Leaf" },
          f1: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
          f2: { kind: "Leaf" },
        },
        f1: { kind: "S", f0: { kind: "Z" } },
        f2: {
          kind: "Node",
          f0: { kind: "Leaf" },
          f1: { kind: "S", f0: { kind: "Z" } },
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
    tree_count_nodes({
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
        f2: {
          kind: "Node",
          f0: { kind: "Leaf" },
          f1: { kind: "S", f0: { kind: "Z" } },
          f2: { kind: "Leaf" },
        },
      },
      f1: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
      f2: {
        kind: "Node",
        f0: {
          kind: "Node",
          f0: { kind: "Leaf" },
          f1: { kind: "S", f0: { kind: "Z" } },
          f2: { kind: "Leaf" },
        },
        f1: { kind: "Z" },
        f2: {
          kind: "Node",
          f0: { kind: "Leaf" },
          f1: { kind: "Z" },
          f2: { kind: "Leaf" },
        },
      },
    }),
    {
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
    }
  );
  assert.deepEqual(
    tree_count_nodes({
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
    }),
    { kind: "S", f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } } }
  );
}
validations();
