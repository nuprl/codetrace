declare var require: any;
const assert = require("node:assert");
type nat = { kind: "Z" } | { kind: "S"; f0: nat };
type nlist = { kind: "Nil" } | { kind: "Cons"; f0: nat; f1: nlist };

function list_make(n: nat): nlist {
  switch (n.kind) {
    case "Z": {
      return { kind: "Nil" };
    }
    case "S": {
      let n1 = n.f0;
      return { kind: "Cons", f0: { kind: "Z" }, f1: list_make(n1) };
    }
  }
}

function assertions() {
  assert.deepEqual(list_make({ kind: "Z" }), { kind: "Nil" });
  assert.deepEqual(
    list_make({
      kind: "S",
      f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
    }),
    {
      kind: "Cons",
      f0: { kind: "Z" },
      f1: {
        kind: "Cons",
        f0: { kind: "Z" },
        f1: { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } },
      },
    }
  );
  assert.deepEqual(
    list_make({ kind: "S", f0: { kind: "S", f0: { kind: "Z" } } }),
    {
      kind: "Cons",
      f0: { kind: "Z" },
      f1: { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } },
    }
  );
}
assertions();

function validations() {
  assert.deepEqual(
    list_make({
      kind: "S",
      f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
    }),
    {
      kind: "Cons",
      f0: { kind: "Z" },
      f1: {
        kind: "Cons",
        f0: { kind: "Z" },
        f1: { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } },
      },
    }
  );
  assert.deepEqual(
    list_make({
      kind: "S",
      f0: {
        kind: "S",
        f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
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
          f0: { kind: "Z" },
          f1: { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } },
        },
      },
    }
  );
}
validations();
