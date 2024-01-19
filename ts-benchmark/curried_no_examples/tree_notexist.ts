declare var require: any;
const assert = require("node:assert");
type nat = { kind: "Z" } | { kind: "S"; f0: nat };
type bbool = { kind: "True" } | { kind: "False" };
type cmp = { kind: "LT" } | { kind: "EQ" } | { kind: "GT" };
type tree = { kind: "Leaf" } | { kind: "Node"; f0: nat; f1: tree; f2: tree };

function compare(x1: nat): (x2: nat) => cmp {
  return function (x2: nat) {
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
            return compare(n1)(n2);
          }
        }
      }
    }
  };
}
function bool_band(b1: bbool): (b2: bbool) => bbool {
  return function (b2: bbool) {
    switch (b1.kind) {
      case "True": {
        return b2;
      }
      case "False": {
        return { kind: "False" };
      }
    }
  };
}

function tree_notexist(n: nat): (t: tree) => bbool {
  return function (t: tree) {
    switch (t.kind) {
      case "Leaf": {
        return { kind: "True" };
      }
      case "Node": {
        let rt = t.f2;
        let lt = t.f1;
        let v = t.f0;
        switch (compare(v)(n).kind) {
          case "EQ": {
            return { kind: "False" };
          }
          case "LT": {
            return bool_band(tree_notexist(n)(lt))(tree_notexist(n)(rt));
          }
          case "GT": {
            return bool_band(tree_notexist(n)(lt))(tree_notexist(n)(rt));
          }
        }
      }
    }
  };
}

function assertions() {
  assert.deepEqual(
    tree_notexist({ kind: "S", f0: { kind: "Z" } })({ kind: "Leaf" }),
    { kind: "True" }
  );
  assert.deepEqual(tree_notexist({ kind: "Z" })({ kind: "Leaf" }), {
    kind: "True",
  });
  assert.deepEqual(
    tree_notexist({ kind: "Z" })({
      kind: "Node",
      f0: { kind: "Z" },
      f1: { kind: "Leaf" },
      f2: { kind: "Leaf" },
    }),
    { kind: "False" }
  );
  assert.deepEqual(
    tree_notexist({ kind: "S", f0: { kind: "Z" } })({
      kind: "Node",
      f0: { kind: "Z" },
      f1: { kind: "Leaf" },
      f2: { kind: "Leaf" },
    }),
    { kind: "True" }
  );
  assert.deepEqual(
    tree_notexist({ kind: "Z" })({
      kind: "Node",
      f0: { kind: "S", f0: { kind: "Z" } },
      f1: { kind: "Leaf" },
      f2: { kind: "Leaf" },
    }),
    { kind: "True" }
  );
  assert.deepEqual(
    tree_notexist({ kind: "Z" })({
      kind: "Node",
      f0: { kind: "S", f0: { kind: "Z" } },
      f1: {
        kind: "Node",
        f0: { kind: "Z" },
        f1: { kind: "Leaf" },
        f2: { kind: "Leaf" },
      },
      f2: { kind: "Leaf" },
    }),
    { kind: "False" }
  );
  assert.deepEqual(
    tree_notexist({ kind: "S", f0: { kind: "Z" } })({
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
    { kind: "False" }
  );
  assert.deepEqual(
    tree_notexist({ kind: "Z" })({
      kind: "Node",
      f0: { kind: "S", f0: { kind: "Z" } },
      f1: { kind: "Leaf" },
      f2: {
        kind: "Node",
        f0: { kind: "Z" },
        f1: { kind: "Leaf" },
        f2: { kind: "Leaf" },
      },
    }),
    { kind: "False" }
  );
  assert.deepEqual(
    tree_notexist({ kind: "S", f0: { kind: "Z" } })({
      kind: "Node",
      f0: { kind: "Z" },
      f1: { kind: "Leaf" },
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
    tree_notexist({ kind: "S", f0: { kind: "Z" } })({
      kind: "Node",
      f0: { kind: "Z" },
      f1: { kind: "Leaf" },
      f2: {
        kind: "Node",
        f0: { kind: "S", f0: { kind: "Z" } },
        f1: { kind: "Leaf" },
        f2: { kind: "Leaf" },
      },
    }),
    { kind: "False" }
  );
  assert.deepEqual(
    tree_notexist({ kind: "S", f0: { kind: "S", f0: { kind: "Z" } } })({
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
    { kind: "False" }
  );
  assert.deepEqual(
    tree_notexist({
      kind: "S",
      f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
    })({
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
    { kind: "True" }
  );
}
assertions();
