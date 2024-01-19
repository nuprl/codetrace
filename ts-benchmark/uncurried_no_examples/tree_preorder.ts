declare var require: any;
const assert = require("node:assert");
type nat = { kind: "Z" } | { kind: "S"; f0: nat };
type nlist = { kind: "Nil" } | { kind: "Cons"; f0: nat; f1: nlist };
type tree = { kind: "Leaf" } | { kind: "Node"; f0: tree; f1: nat; f2: tree };

function append(l1: nlist, l2: nlist): nlist {
  switch (l1.kind) {
    case "Nil": {
      return l2;
    }
    case "Cons": {
      let xs = l1.f1;
      let x = l1.f0;
      return { kind: "Cons", f0: x, f1: append(xs, l2) };
    }
  }
}

function tree_preorder(t: tree): nlist {
  switch (t.kind) {
    case "Leaf": {
      return { kind: "Nil" };
    }
    case "Node": {
      let r = t.f2;
      let n = t.f1;
      let l = t.f0;
      return append(
        { kind: "Cons", f0: n, f1: tree_preorder(l) },
        tree_preorder(r)
      );
    }
  }
}

function assertions() {
  assert.deepEqual(tree_preorder({ kind: "Leaf" }), { kind: "Nil" });
  assert.deepEqual(
    tree_preorder({
      kind: "Node",
      f0: {
        kind: "Node",
        f0: { kind: "Leaf" },
        f1: { kind: "S", f0: { kind: "Z" } },
        f2: { kind: "Leaf" },
      },
      f1: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
      f2: { kind: "Leaf" },
    }),
    {
      kind: "Cons",
      f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
      f1: {
        kind: "Cons",
        f0: { kind: "S", f0: { kind: "Z" } },
        f1: { kind: "Nil" },
      },
    }
  );
  assert.deepEqual(
    tree_preorder({
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
      kind: "Cons",
      f0: { kind: "S", f0: { kind: "Z" } },
      f1: {
        kind: "Cons",
        f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
        f1: { kind: "Nil" },
      },
    }
  );
}
assertions();

function validations() {
  assert.deepEqual(
    tree_preorder({
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
          f1: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
          f2: { kind: "Leaf" },
        },
      },
      f1: {
        kind: "S",
        f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
      },
      f2: {
        kind: "Node",
        f0: {
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
        f2: {
          kind: "Node",
          f0: { kind: "Leaf" },
          f1: {
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
          f2: { kind: "Leaf" },
        },
      },
    }),
    {
      kind: "Cons",
      f0: {
        kind: "S",
        f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
      },
      f1: {
        kind: "Cons",
        f0: { kind: "S", f0: { kind: "Z" } },
        f1: {
          kind: "Cons",
          f0: { kind: "Z" },
          f1: {
            kind: "Cons",
            f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
            f1: {
              kind: "Cons",
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
              f1: {
                kind: "Cons",
                f0: {
                  kind: "S",
                  f0: {
                    kind: "S",
                    f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
                  },
                },
                f1: {
                  kind: "Cons",
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
                            f0: { kind: "S", f0: { kind: "Z" } },
                          },
                        },
                      },
                    },
                  },
                  f1: { kind: "Nil" },
                },
              },
            },
          },
        },
      },
    }
  );
  assert.deepEqual(
    tree_preorder({
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
    }),
    {
      kind: "Cons",
      f0: { kind: "Z" },
      f1: {
        kind: "Cons",
        f0: { kind: "S", f0: { kind: "Z" } },
        f1: {
          kind: "Cons",
          f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
          f1: { kind: "Nil" },
        },
      },
    }
  );
  assert.deepEqual(
    tree_preorder({
      kind: "Node",
      f0: {
        kind: "Node",
        f0: {
          kind: "Node",
          f0: { kind: "Leaf" },
          f1: { kind: "S", f0: { kind: "Z" } },
          f2: { kind: "Leaf" },
        },
        f1: { kind: "Z" },
        f2: { kind: "Leaf" },
      },
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
    }),
    {
      kind: "Cons",
      f0: { kind: "Z" },
      f1: {
        kind: "Cons",
        f0: { kind: "Z" },
        f1: {
          kind: "Cons",
          f0: { kind: "S", f0: { kind: "Z" } },
          f1: {
            kind: "Cons",
            f0: { kind: "S", f0: { kind: "Z" } },
            f1: {
              kind: "Cons",
              f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
              f1: { kind: "Nil" },
            },
          },
        },
      },
    }
  );
}
validations();
