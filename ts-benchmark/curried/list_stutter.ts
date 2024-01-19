declare var require: any;
const assert = require("node:assert");
type nat = { kind: "Z" } | { kind: "S"; f0: nat };
type nlist = { kind: "Nil" } | { kind: "Cons"; f0: nat; f1: nlist };

function list_stutter(lox: nlist): nlist {
  switch (lox.kind) {
    case "Nil": {
      return { kind: "Nil" };
    }
    case "Cons": {
      let xs = lox.f1;
      let x = lox.f0;
      return {
        kind: "Cons",
        f0: x,
        f1: { kind: "Cons", f0: x, f1: list_stutter(xs) },
      };
    }
  }
}

function assertions() {
  assert.deepEqual(list_stutter({ kind: "Nil" }), { kind: "Nil" });
  assert.deepEqual(
    list_stutter({
      kind: "Cons",
      f0: { kind: "S", f0: { kind: "Z" } },
      f1: { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } },
    }),
    {
      kind: "Cons",
      f0: { kind: "S", f0: { kind: "Z" } },
      f1: {
        kind: "Cons",
        f0: { kind: "S", f0: { kind: "Z" } },
        f1: {
          kind: "Cons",
          f0: { kind: "Z" },
          f1: { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } },
        },
      },
    }
  );
}
assertions();

function validations() {
  assert.deepEqual(
    list_stutter({ kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } }),
    {
      kind: "Cons",
      f0: { kind: "Z" },
      f1: { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } },
    }
  );
  assert.deepEqual(
    list_stutter({
      kind: "Cons",
      f0: { kind: "Z" },
      f1: {
        kind: "Cons",
        f0: { kind: "S", f0: { kind: "Z" } },
        f1: { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } },
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
              f0: { kind: "Z" },
              f1: { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } },
            },
          },
        },
      },
    }
  );
}
validations();
