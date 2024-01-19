declare var require: any;
const assert = require("node:assert");
type nat = { kind: "Z" } | { kind: "S"; f0: nat };
type nlist = { kind: "Nil" } | { kind: "Cons"; f0: nat; f1: nlist };

function fold(l: nlist, f: (__x7: nat, __x8: nat) => nat, acc: nat): nat {
  switch (l.kind) {
    case "Nil": {
      return acc;
    }
    case "Cons": {
      let xs = l.f1;
      let x = l.f0;
      return fold(xs, f, f(acc, x));
    }
  }
}
function add(n1: nat, n2: nat): nat {
  switch (n1.kind) {
    case "Z": {
      return n2;
    }
    case "S": {
      let n1p = n1.f0;
      return { kind: "S", f0: add(n1p, n2) };
    }
  }
}

function list_sum(l: nlist): nat {
  return fold(l, add, { kind: "Z" });
}

function assertions() {
  assert.deepEqual(list_sum({ kind: "Nil" }), { kind: "Z" });
  assert.deepEqual(
    list_sum({
      kind: "Cons",
      f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
      f1: {
        kind: "Cons",
        f0: { kind: "S", f0: { kind: "Z" } },
        f1: { kind: "Nil" },
      },
    }),
    { kind: "S", f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } } }
  );
}
assertions();

function validations() {
  assert.deepEqual(
    list_sum({ kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } }),
    { kind: "Z" }
  );
  assert.deepEqual(
    list_sum({
      kind: "Cons",
      f0: { kind: "S", f0: { kind: "Z" } },
      f1: {
        kind: "Cons",
        f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
        f1: { kind: "Nil" },
      },
    }),
    { kind: "S", f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } } }
  );
  assert.deepEqual(
    list_sum({
      kind: "Cons",
      f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
      f1: {
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
                f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
              },
            },
          },
          f1: { kind: "Nil" },
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
              f0: {
                kind: "S",
                f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
              },
            },
          },
        },
      },
    }
  );
}
validations();
