declare var require: any;
const assert = require("node:assert");
type bbool = { kind: "True" } | { kind: "False" };
type tree = { kind: "Leaf" } | { kind: "Node"; f0: tree; f1: bbool; f2: tree };
type blist = { kind: "Nil" } | { kind: "Cons"; f0: bbool; f1: blist };

function list_append(l1: blist): (l2: blist) => blist {
  return function (l2: blist) {
    switch (l1.kind) {
      case "Nil": {
        return l2;
      }
      case "Cons": {
        let xs = l1.f1;
        let x = l1.f0;
        return { kind: "Cons", f0: x, f1: list_append(xs)(l2) };
      }
    }
  };
}

function tree_collect_leaves(t: tree): blist {
  switch (t.kind) {
    case "Leaf": {
      return { kind: "Nil" };
    }
    case "Node": {
      let rt = t.f2;
      let b = t.f1;
      let lt = t.f0;
      switch (lt.kind) {
        case "Leaf": {
          switch (rt.kind) {
            case "Leaf": {
              return { kind: "Cons", f0: b, f1: { kind: "Nil" } };
            }
            case "Node": {
              let rrt = rt.f2;
              let rb = rt.f1;
              let rlt = rt.f0;
              return { kind: "Cons", f0: b, f1: tree_collect_leaves(rt) };
            }
          }
        }
        case "Node": {
          let lrt = lt.f2;
          let lb = lt.f1;
          let llt = lt.f0;
          switch (rt.kind) {
            case "Leaf": {
              return list_append(tree_collect_leaves(lt))({
                kind: "Cons",
                f0: b,
                f1: { kind: "Nil" },
              });
            }
            case "Node": {
              let rrt = rt.f2;
              let rb = rt.f1;
              let rlt = rt.f0;
              return list_append(tree_collect_leaves(lt))(
                tree_collect_leaves(rt)
              );
            }
          }
        }
      }
    }
  }
}

function assertions() {
  assert.deepEqual(tree_collect_leaves({ kind: "Leaf" }), { kind: "Nil" });
  assert.deepEqual(
    tree_collect_leaves({
      kind: "Node",
      f0: {
        kind: "Node",
        f0: { kind: "Leaf" },
        f1: { kind: "True" },
        f2: { kind: "Leaf" },
      },
      f1: { kind: "False" },
      f2: { kind: "Leaf" },
    }),
    {
      kind: "Cons",
      f0: { kind: "True" },
      f1: { kind: "Cons", f0: { kind: "False" }, f1: { kind: "Nil" } },
    }
  );
  assert.deepEqual(
    tree_collect_leaves({
      kind: "Node",
      f0: { kind: "Leaf" },
      f1: { kind: "False" },
      f2: {
        kind: "Node",
        f0: { kind: "Leaf" },
        f1: { kind: "True" },
        f2: { kind: "Leaf" },
      },
    }),
    {
      kind: "Cons",
      f0: { kind: "False" },
      f1: { kind: "Cons", f0: { kind: "True" }, f1: { kind: "Nil" } },
    }
  );
}
assertions();

function validations() {
  assert.deepEqual(
    tree_collect_leaves({
      kind: "Node",
      f0: { kind: "Leaf" },
      f1: { kind: "False" },
      f2: { kind: "Leaf" },
    }),
    { kind: "Cons", f0: { kind: "False" }, f1: { kind: "Nil" } }
  );
  assert.deepEqual(
    tree_collect_leaves({
      kind: "Node",
      f0: {
        kind: "Node",
        f0: { kind: "Leaf" },
        f1: { kind: "True" },
        f2: {
          kind: "Node",
          f0: { kind: "Leaf" },
          f1: { kind: "False" },
          f2: { kind: "Leaf" },
        },
      },
      f1: { kind: "True" },
      f2: {
        kind: "Node",
        f0: { kind: "Leaf" },
        f1: { kind: "False" },
        f2: { kind: "Leaf" },
      },
    }),
    {
      kind: "Cons",
      f0: { kind: "True" },
      f1: {
        kind: "Cons",
        f0: { kind: "False" },
        f1: { kind: "Cons", f0: { kind: "False" }, f1: { kind: "Nil" } },
      },
    }
  );
  assert.deepEqual(
    tree_collect_leaves({
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
        f2: {
          kind: "Node",
          f0: {
            kind: "Node",
            f0: { kind: "Leaf" },
            f1: { kind: "False" },
            f2: { kind: "Leaf" },
          },
          f1: { kind: "True" },
          f2: { kind: "Leaf" },
        },
      },
      f1: { kind: "True" },
      f2: {
        kind: "Node",
        f0: { kind: "Leaf" },
        f1: { kind: "False" },
        f2: {
          kind: "Node",
          f0: { kind: "Leaf" },
          f1: { kind: "True" },
          f2: { kind: "Leaf" },
        },
      },
    }),
    {
      kind: "Cons",
      f0: { kind: "True" },
      f1: {
        kind: "Cons",
        f0: { kind: "False" },
        f1: {
          kind: "Cons",
          f0: { kind: "True" },
          f1: {
            kind: "Cons",
            f0: { kind: "False" },
            f1: { kind: "Cons", f0: { kind: "True" }, f1: { kind: "Nil" } },
          },
        },
      },
    }
  );
}
validations();
