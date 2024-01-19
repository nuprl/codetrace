declare var require: any;
const assert = require("node:assert");
type nat = { kind: "Z" } | { kind: "S"; f0: nat };
type nlist = { kind: "Nil" } | { kind: "Cons"; f0: nat; f1: nlist };

function list_length(lox: nlist): nat {
  switch (lox.kind) {
    case "Nil": {
      return { kind: "Z" };
    }
    case "Cons": {
      let xs = lox.f1;
      let n = lox.f0;
      return { kind: "S", f0: list_length(xs) };
    }
  }
}

function assertions() {
  assert.deepEqual(list_length({ kind: "Nil" }), { kind: "Z" });
  assert.deepEqual(
    list_length({ kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } }),
    { kind: "S", f0: { kind: "Z" } }
  );
  assert.deepEqual(
    list_length({
      kind: "Cons",
      f0: { kind: "Z" },
      f1: { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } },
    }),
    { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } }
  );
}
assertions();

function validations() {
  assert.deepEqual(
    list_length({
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
    { kind: "S", f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } } }
  );
  assert.deepEqual(
    list_length({
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
      kind: "S",
      f0: {
        kind: "S",
        f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
      },
    }
  );
}
validations();
