declare var require: any;
const assert = require("node:assert");
type nat = { kind: "Z" } | { kind: "S"; f0: nat };
type nlist = { kind: "Nil" } | { kind: "Cons"; f0: nat; f1: nlist };

function list_take(n: nat, lox: nlist): nlist {
  switch (n.kind) {
    case "Z": {
      return { kind: "Nil" };
    }
    case "S": {
      let np = n.f0;
      switch (lox.kind) {
        case "Nil": {
          return { kind: "Nil" };
        }
        case "Cons": {
          let xs = lox.f1;
          let x = lox.f0;
          return { kind: "Cons", f0: x, f1: list_take(np, xs) };
        }
      }
    }
  }
}

function assertions() {
  assert.deepEqual(list_take({ kind: "Z" }, { kind: "Nil" }), { kind: "Nil" });
  assert.deepEqual(
    list_take(
      { kind: "Z" },
      {
        kind: "Cons",
        f0: { kind: "Z" },
        f1: {
          kind: "Cons",
          f0: { kind: "S", f0: { kind: "Z" } },
          f1: { kind: "Nil" },
        },
      }
    ),
    { kind: "Nil" }
  );
  assert.deepEqual(
    list_take(
      { kind: "S", f0: { kind: "Z" } },
      {
        kind: "Cons",
        f0: { kind: "S", f0: { kind: "Z" } },
        f1: { kind: "Nil" },
      }
    ),
    { kind: "Cons", f0: { kind: "S", f0: { kind: "Z" } }, f1: { kind: "Nil" } }
  );
  assert.deepEqual(
    list_take(
      { kind: "S", f0: { kind: "Z" } },
      {
        kind: "Cons",
        f0: { kind: "Z" },
        f1: {
          kind: "Cons",
          f0: { kind: "S", f0: { kind: "Z" } },
          f1: { kind: "Nil" },
        },
      }
    ),
    { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } }
  );
  assert.deepEqual(
    list_take(
      { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
      {
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
      }
    ),
    {
      kind: "Cons",
      f0: { kind: "S", f0: { kind: "Z" } },
      f1: { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } },
    }
  );
}
assertions();

function validations() {
  assert.deepEqual(
    list_take(
      { kind: "Z" },
      {
        kind: "Cons",
        f0: { kind: "S", f0: { kind: "Z" } },
        f1: { kind: "Nil" },
      }
    ),
    { kind: "Nil" }
  );
  assert.deepEqual(
    list_take(
      { kind: "S", f0: { kind: "Z" } },
      { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } }
    ),
    { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } }
  );
  assert.deepEqual(
    list_take(
      { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
      { kind: "Nil" }
    ),
    { kind: "Nil" }
  );
  assert.deepEqual(
    list_take(
      { kind: "S", f0: { kind: "Z" } },
      {
        kind: "Cons",
        f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
        f1: { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } },
      }
    ),
    {
      kind: "Cons",
      f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
      f1: { kind: "Nil" },
    }
  );
  assert.deepEqual(
    list_take(
      { kind: "S", f0: { kind: "Z" } },
      {
        kind: "Cons",
        f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
        f1: { kind: "Nil" },
      }
    ),
    {
      kind: "Cons",
      f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
      f1: { kind: "Nil" },
    }
  );
}
validations();
