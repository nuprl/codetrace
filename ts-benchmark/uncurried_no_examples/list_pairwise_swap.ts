declare var require: any;
const assert = require("node:assert");
type nat = { kind: "Z" } | { kind: "S"; f0: nat };
type nlist = { kind: "Nil" } | { kind: "Cons"; f0: nat; f1: nlist };

function list_pairwise_swap(l: nlist): nlist {
  switch (l.kind) {
    case "Nil": {
      return { kind: "Nil" };
    }
    case "Cons": {
      let l2 = l.f1;
      let n1 = l.f0;
      switch (l2.kind) {
        case "Nil": {
          return { kind: "Nil" };
        }
        case "Cons": {
          let l3 = l2.f1;
          let n2 = l2.f0;
          return {
            kind: "Cons",
            f0: n2,
            f1: { kind: "Cons", f0: n1, f1: list_pairwise_swap(l3) },
          };
        }
      }
    }
  }
}

function assertions() {
  assert.deepEqual(list_pairwise_swap({ kind: "Nil" }), { kind: "Nil" });
  assert.deepEqual(
    list_pairwise_swap({
      kind: "Cons",
      f0: { kind: "S", f0: { kind: "Z" } },
      f1: { kind: "Nil" },
    }),
    { kind: "Nil" }
  );
  assert.deepEqual(
    list_pairwise_swap({
      kind: "Cons",
      f0: { kind: "S", f0: { kind: "Z" } },
      f1: { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } },
    }),
    {
      kind: "Cons",
      f0: { kind: "Z" },
      f1: {
        kind: "Cons",
        f0: { kind: "S", f0: { kind: "Z" } },
        f1: { kind: "Nil" },
      },
    }
  );
  assert.deepEqual(
    list_pairwise_swap({
      kind: "Cons",
      f0: { kind: "S", f0: { kind: "Z" } },
      f1: {
        kind: "Cons",
        f0: { kind: "Z" },
        f1: {
          kind: "Cons",
          f0: { kind: "S", f0: { kind: "Z" } },
          f1: { kind: "Nil" },
        },
      },
    }),
    {
      kind: "Cons",
      f0: { kind: "Z" },
      f1: {
        kind: "Cons",
        f0: { kind: "S", f0: { kind: "Z" } },
        f1: { kind: "Nil" },
      },
    }
  );
  assert.deepEqual(
    list_pairwise_swap({
      kind: "Cons",
      f0: { kind: "Z" },
      f1: {
        kind: "Cons",
        f0: { kind: "S", f0: { kind: "Z" } },
        f1: {
          kind: "Cons",
          f0: { kind: "Z" },
          f1: {
            kind: "Cons",
            f0: { kind: "S", f0: { kind: "Z" } },
            f1: { kind: "Nil" },
          },
        },
      },
    }),
    {
      kind: "Cons",
      f0: { kind: "S", f0: { kind: "Z" } },
      f1: {
        kind: "Cons",
        f0: { kind: "Z" },
        f1: {
          kind: "Cons",
          f0: { kind: "S", f0: { kind: "Z" } },
          f1: { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } },
        },
      },
    }
  );
}
assertions();

function validations() {
  assert.deepEqual(
    list_pairwise_swap({
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
        f0: { kind: "S", f0: { kind: "Z" } },
        f1: {
          kind: "Cons",
          f0: {
            kind: "S",
            f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
          },
          f1: { kind: "Nil" },
        },
      },
    }),
    {
      kind: "Cons",
      f0: { kind: "S", f0: { kind: "Z" } },
      f1: {
        kind: "Cons",
        f0: {
          kind: "S",
          f0: {
            kind: "S",
            f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
          },
        },
        f1: { kind: "Nil" },
      },
    }
  );
  assert.deepEqual(
    list_pairwise_swap({
      kind: "Cons",
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
      f1: {
        kind: "Cons",
        f0: { kind: "S", f0: { kind: "Z" } },
        f1: {
          kind: "Cons",
          f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
          f1: { kind: "Nil" },
        },
      },
    }),
    {
      kind: "Cons",
      f0: { kind: "S", f0: { kind: "Z" } },
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
                f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
              },
            },
          },
        },
        f1: { kind: "Nil" },
      },
    }
  );
  assert.deepEqual(
    list_pairwise_swap({
      kind: "Cons",
      f0: { kind: "S", f0: { kind: "Z" } },
      f1: {
        kind: "Cons",
        f0: { kind: "S", f0: { kind: "Z" } },
        f1: {
          kind: "Cons",
          f0: { kind: "S", f0: { kind: "Z" } },
          f1: { kind: "Nil" },
        },
      },
    }),
    {
      kind: "Cons",
      f0: { kind: "S", f0: { kind: "Z" } },
      f1: {
        kind: "Cons",
        f0: { kind: "S", f0: { kind: "Z" } },
        f1: { kind: "Nil" },
      },
    }
  );
  assert.deepEqual(
    list_pairwise_swap({
      kind: "Cons",
      f0: { kind: "S", f0: { kind: "Z" } },
      f1: {
        kind: "Cons",
        f0: { kind: "S", f0: { kind: "Z" } },
        f1: {
          kind: "Cons",
          f0: { kind: "S", f0: { kind: "Z" } },
          f1: {
            kind: "Cons",
            f0: { kind: "S", f0: { kind: "Z" } },
            f1: { kind: "Nil" },
          },
        },
      },
    }),
    {
      kind: "Cons",
      f0: { kind: "S", f0: { kind: "Z" } },
      f1: {
        kind: "Cons",
        f0: { kind: "S", f0: { kind: "Z" } },
        f1: {
          kind: "Cons",
          f0: { kind: "S", f0: { kind: "Z" } },
          f1: {
            kind: "Cons",
            f0: { kind: "S", f0: { kind: "Z" } },
            f1: { kind: "Nil" },
          },
        },
      },
    }
  );
}
validations();
