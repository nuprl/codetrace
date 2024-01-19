declare var require: any;
const assert = require("node:assert");
type mybool = { kind: "True" } | { kind: "False" };
type tree = { kind: "Leaf" } | { kind: "Node"; f0: tree; f1: mybool; f2: tree };
type nat = { kind: "Z" } | { kind: "S"; f0: nat };

function sum(n1: nat, n2: nat): nat {
  switch (n1.kind) {
    case "Z": {
      return n2;
    }
    case "S": {
      let n3 = n1.f0;
      return { kind: "S", f0: sum(n3, n2) };
    }
  }
}

function tree_count_leaves(t: tree): nat {
  switch (t.kind) {
    case "Leaf": {
      return { kind: "S", f0: { kind: "Z" } };
    }
    case "Node": {
      let r = t.f2;
      let data = t.f1;
      let l = t.f0;
      return sum(tree_count_leaves(l), tree_count_leaves(r));
    }
  }
}

function assertions() {
  assert.deepEqual(tree_count_leaves({ kind: "Leaf" }), {
    kind: "S",
    f0: { kind: "Z" },
  });
  assert.deepEqual(
    tree_count_leaves({
      kind: "Node",
      f0: {
        kind: "Node",
        f0: { kind: "Leaf" },
        f1: { kind: "True" },
        f2: { kind: "Leaf" },
      },
      f1: { kind: "True" },
      f2: { kind: "Leaf" },
    }),
    { kind: "S", f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } } }
  );
  assert.deepEqual(
    tree_count_leaves({
      kind: "Node",
      f0: {
        kind: "Node",
        f0: { kind: "Leaf" },
        f1: { kind: "True" },
        f2: { kind: "Leaf" },
      },
      f1: { kind: "True" },
      f2: {
        kind: "Node",
        f0: { kind: "Leaf" },
        f1: { kind: "True" },
        f2: { kind: "Leaf" },
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
}
assertions();

function validations() {
  assert.deepEqual(
    tree_count_leaves({
      kind: "Node",
      f0: { kind: "Leaf" },
      f1: { kind: "True" },
      f2: { kind: "Leaf" },
    }),
    { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } }
  );
  assert.deepEqual(
    tree_count_leaves({
      kind: "Node",
      f0: {
        kind: "Node",
        f0: {
          kind: "Node",
          f0: { kind: "Leaf" },
          f1: { kind: "True" },
          f2: { kind: "Leaf" },
        },
        f1: { kind: "False" },
        f2: { kind: "Leaf" },
      },
      f1: { kind: "False" },
      f2: {
        kind: "Node",
        f0: { kind: "Leaf" },
        f1: { kind: "False" },
        f2: {
          kind: "Node",
          f0: { kind: "Leaf" },
          f1: { kind: "False" },
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
            f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
          },
        },
      },
    }
  );
  assert.deepEqual(
    tree_count_leaves({
      kind: "Node",
      f0: { kind: "Leaf" },
      f1: { kind: "True" },
      f2: {
        kind: "Node",
        f0: { kind: "Leaf" },
        f1: { kind: "False" },
        f2: { kind: "Leaf" },
      },
    }),
    { kind: "S", f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } } }
  );
}
validations();
