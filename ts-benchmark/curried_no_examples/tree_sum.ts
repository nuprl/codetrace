declare var require: any;
const assert = require("node:assert");
type nat = { kind: "Z" } | { kind: "S"; f0: nat };
type tree = { kind: "Leaf" } | { kind: "Node"; f0: nat; f1: tree; f2: tree };

function add(n1: nat): (n2: nat) => nat {
  return function (n2: nat) {
    switch (n1.kind) {
      case "Z": {
        return n2;
      }
      case "S": {
        let n3 = n1.f0;
        return { kind: "S", f0: add(n3)(n2) };
      }
    }
  };
}

function tree_sum(t: tree): nat {
  switch (t.kind) {
    case "Leaf": {
      return { kind: "Z" };
    }
    case "Node": {
      let rt = t.f2;
      let lt = t.f1;
      let n = t.f0;
      return add(add(n)(tree_sum(lt)))(tree_sum(rt));
    }
  }
}

function assertions() {
  assert.deepEqual(tree_sum({ kind: "Leaf" }), { kind: "Z" });
  assert.deepEqual(
    tree_sum({
      kind: "Node",
      f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
      f1: { kind: "Leaf" },
      f2: { kind: "Leaf" },
    }),
    { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } }
  );
  assert.deepEqual(
    tree_sum({
      kind: "Node",
      f0: { kind: "S", f0: { kind: "Z" } },
      f1: {
        kind: "Node",
        f0: { kind: "S", f0: { kind: "Z" } },
        f1: { kind: "Leaf" },
        f2: { kind: "Leaf" },
      },
      f2: { kind: "Leaf" },
    }),
    { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } }
  );
  assert.deepEqual(
    tree_sum({
      kind: "Node",
      f0: { kind: "S", f0: { kind: "Z" } },
      f1: { kind: "Leaf" },
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
    tree_sum({
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
    { kind: "S", f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } } }
  );
  assert.deepEqual(
    tree_sum({
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
    tree_sum({
      kind: "Node",
      f0: { kind: "Z" },
      f1: {
        kind: "Node",
        f0: { kind: "Z" },
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
      f2: {
        kind: "Node",
        f0: { kind: "Z" },
        f1: { kind: "Leaf" },
        f2: { kind: "Leaf" },
      },
    }),
    { kind: "Z" }
  );
  assert.deepEqual(
    tree_sum({
      kind: "Node",
      f0: { kind: "S", f0: { kind: "Z" } },
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
    {
      kind: "S",
      f0: {
        kind: "S",
        f0: {
          kind: "S",
          f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
        },
      },
    }
  );
  assert.deepEqual(
    tree_sum({
      kind: "Node",
      f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
      f1: { kind: "Leaf" },
      f2: {
        kind: "Node",
        f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
        f1: {
          kind: "Node",
          f0: { kind: "Z" },
          f1: { kind: "Leaf" },
          f2: { kind: "Leaf" },
        },
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
validations();
