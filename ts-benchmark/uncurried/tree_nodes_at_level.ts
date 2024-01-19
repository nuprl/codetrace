declare var require: any;
const assert = require("node:assert");
type bbool = { kind: "True" } | { kind: "False" };
type nat = { kind: "Z" } | { kind: "S"; f0: nat };
type tree = { kind: "Leaf" } | { kind: "Node"; f0: tree; f1: bbool; f2: tree };

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

function tree_nodes_at_level(t: tree, n: nat): nat {
  switch (t.kind) {
    case "Leaf": {
      return { kind: "Z" };
    }
    case "Node": {
      let r = t.f2;
      let d = t.f1;
      let l = t.f0;
      switch (n.kind) {
        case "Z": {
          return { kind: "S", f0: { kind: "Z" } };
        }
        case "S": {
          let np = n.f0;
          return sum(tree_nodes_at_level(l, np), tree_nodes_at_level(r, np));
        }
      }
    }
  }
}

function assertions() {
  assert.deepEqual(tree_nodes_at_level({ kind: "Leaf" }, { kind: "Z" }), {
    kind: "Z",
  });
  assert.deepEqual(
    tree_nodes_at_level({ kind: "Leaf" }, { kind: "S", f0: { kind: "Z" } }),
    { kind: "Z" }
  );
  assert.deepEqual(
    tree_nodes_at_level(
      { kind: "Leaf" },
      { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } }
    ),
    { kind: "Z" }
  );
  assert.deepEqual(
    tree_nodes_at_level(
      { kind: "Leaf" },
      { kind: "S", f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } } }
    ),
    { kind: "Z" }
  );
  assert.deepEqual(
    tree_nodes_at_level(
      {
        kind: "Node",
        f0: { kind: "Leaf" },
        f1: { kind: "True" },
        f2: { kind: "Leaf" },
      },
      { kind: "Z" }
    ),
    { kind: "S", f0: { kind: "Z" } }
  );
  assert.deepEqual(
    tree_nodes_at_level(
      {
        kind: "Node",
        f0: { kind: "Leaf" },
        f1: { kind: "True" },
        f2: { kind: "Leaf" },
      },
      { kind: "S", f0: { kind: "Z" } }
    ),
    { kind: "Z" }
  );
  assert.deepEqual(
    tree_nodes_at_level(
      {
        kind: "Node",
        f0: { kind: "Leaf" },
        f1: { kind: "True" },
        f2: { kind: "Leaf" },
      },
      { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } }
    ),
    { kind: "Z" }
  );
  assert.deepEqual(
    tree_nodes_at_level(
      {
        kind: "Node",
        f0: { kind: "Leaf" },
        f1: { kind: "True" },
        f2: { kind: "Leaf" },
      },
      { kind: "S", f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } } }
    ),
    { kind: "Z" }
  );
  assert.deepEqual(
    tree_nodes_at_level(
      {
        kind: "Node",
        f0: {
          kind: "Node",
          f0: { kind: "Leaf" },
          f1: { kind: "True" },
          f2: { kind: "Leaf" },
        },
        f1: { kind: "True" },
        f2: { kind: "Leaf" },
      },
      { kind: "Z" }
    ),
    { kind: "S", f0: { kind: "Z" } }
  );
  assert.deepEqual(
    tree_nodes_at_level(
      {
        kind: "Node",
        f0: {
          kind: "Node",
          f0: { kind: "Leaf" },
          f1: { kind: "True" },
          f2: { kind: "Leaf" },
        },
        f1: { kind: "True" },
        f2: { kind: "Leaf" },
      },
      { kind: "S", f0: { kind: "Z" } }
    ),
    { kind: "S", f0: { kind: "Z" } }
  );
  assert.deepEqual(
    tree_nodes_at_level(
      {
        kind: "Node",
        f0: {
          kind: "Node",
          f0: { kind: "Leaf" },
          f1: { kind: "True" },
          f2: { kind: "Leaf" },
        },
        f1: { kind: "True" },
        f2: { kind: "Leaf" },
      },
      { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } }
    ),
    { kind: "Z" }
  );
  assert.deepEqual(
    tree_nodes_at_level(
      {
        kind: "Node",
        f0: {
          kind: "Node",
          f0: { kind: "Leaf" },
          f1: { kind: "True" },
          f2: { kind: "Leaf" },
        },
        f1: { kind: "True" },
        f2: { kind: "Leaf" },
      },
      { kind: "S", f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } } }
    ),
    { kind: "Z" }
  );
  assert.deepEqual(
    tree_nodes_at_level(
      {
        kind: "Node",
        f0: { kind: "Leaf" },
        f1: { kind: "True" },
        f2: {
          kind: "Node",
          f0: { kind: "Leaf" },
          f1: { kind: "True" },
          f2: { kind: "Leaf" },
        },
      },
      { kind: "Z" }
    ),
    { kind: "S", f0: { kind: "Z" } }
  );
  assert.deepEqual(
    tree_nodes_at_level(
      {
        kind: "Node",
        f0: { kind: "Leaf" },
        f1: { kind: "True" },
        f2: {
          kind: "Node",
          f0: { kind: "Leaf" },
          f1: { kind: "True" },
          f2: { kind: "Leaf" },
        },
      },
      { kind: "S", f0: { kind: "Z" } }
    ),
    { kind: "S", f0: { kind: "Z" } }
  );
  assert.deepEqual(
    tree_nodes_at_level(
      {
        kind: "Node",
        f0: { kind: "Leaf" },
        f1: { kind: "True" },
        f2: {
          kind: "Node",
          f0: { kind: "Leaf" },
          f1: { kind: "True" },
          f2: { kind: "Leaf" },
        },
      },
      { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } }
    ),
    { kind: "Z" }
  );
  assert.deepEqual(
    tree_nodes_at_level(
      {
        kind: "Node",
        f0: { kind: "Leaf" },
        f1: { kind: "True" },
        f2: {
          kind: "Node",
          f0: { kind: "Leaf" },
          f1: { kind: "True" },
          f2: { kind: "Leaf" },
        },
      },
      { kind: "S", f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } } }
    ),
    { kind: "Z" }
  );
  assert.deepEqual(
    tree_nodes_at_level(
      {
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
      },
      { kind: "Z" }
    ),
    { kind: "S", f0: { kind: "Z" } }
  );
  assert.deepEqual(
    tree_nodes_at_level(
      {
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
      },
      { kind: "S", f0: { kind: "Z" } }
    ),
    { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } }
  );
  assert.deepEqual(
    tree_nodes_at_level(
      {
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
      },
      { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } }
    ),
    { kind: "Z" }
  );
  assert.deepEqual(
    tree_nodes_at_level(
      {
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
      },
      { kind: "S", f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } } }
    ),
    { kind: "Z" }
  );
  assert.deepEqual(
    tree_nodes_at_level(
      {
        kind: "Node",
        f0: {
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
        },
        f1: { kind: "True" },
        f2: { kind: "Leaf" },
      },
      { kind: "Z" }
    ),
    { kind: "S", f0: { kind: "Z" } }
  );
  assert.deepEqual(
    tree_nodes_at_level(
      {
        kind: "Node",
        f0: {
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
        },
        f1: { kind: "True" },
        f2: { kind: "Leaf" },
      },
      { kind: "S", f0: { kind: "Z" } }
    ),
    { kind: "S", f0: { kind: "Z" } }
  );
  assert.deepEqual(
    tree_nodes_at_level(
      {
        kind: "Node",
        f0: {
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
        },
        f1: { kind: "True" },
        f2: { kind: "Leaf" },
      },
      { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } }
    ),
    { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } }
  );
  assert.deepEqual(
    tree_nodes_at_level(
      {
        kind: "Node",
        f0: {
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
        },
        f1: { kind: "True" },
        f2: { kind: "Leaf" },
      },
      { kind: "S", f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } } }
    ),
    { kind: "Z" }
  );
}
assertions();
