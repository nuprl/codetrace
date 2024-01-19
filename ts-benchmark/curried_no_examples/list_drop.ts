declare var require: any;
const assert = require("node:assert");
type nat = { kind: "Z" } | { kind: "S"; f0: nat };
type nlist = { kind: "Nil" } | { kind: "Cons"; f0: nat; f1: nlist };

function list_drop(lox: nlist): (n: nat) => nlist {
  return function (n: nat) {
    switch (n.kind) {
      case "Z": {
        return lox;
      }
      case "S": {
        let n1 = n.f0;
        switch (lox.kind) {
          case "Nil": {
            return { kind: "Nil" };
          }
          case "Cons": {
            let xs = lox.f1;
            let x = lox.f0;
            return list_drop(xs)(n1);
          }
        }
      }
    }
  };
}

function assertions() {
  assert.deepEqual(list_drop({ kind: "Nil" })({ kind: "Z" }), { kind: "Nil" });
  assert.deepEqual(
    list_drop({
      kind: "Cons",
      f0: { kind: "S", f0: { kind: "Z" } },
      f1: { kind: "Nil" },
    })({ kind: "Z" }),
    { kind: "Cons", f0: { kind: "S", f0: { kind: "Z" } }, f1: { kind: "Nil" } }
  );
  assert.deepEqual(
    list_drop({
      kind: "Cons",
      f0: { kind: "S", f0: { kind: "Z" } },
      f1: { kind: "Nil" },
    })({ kind: "S", f0: { kind: "Z" } }),
    { kind: "Nil" }
  );
  assert.deepEqual(
    list_drop({
      kind: "Cons",
      f0: { kind: "S", f0: { kind: "Z" } },
      f1: { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } },
    })({ kind: "S", f0: { kind: "Z" } }),
    { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } }
  );
  assert.deepEqual(
    list_drop({
      kind: "Cons",
      f0: { kind: "Z" },
      f1: {
        kind: "Cons",
        f0: { kind: "S", f0: { kind: "Z" } },
        f1: { kind: "Nil" },
      },
    })({ kind: "S", f0: { kind: "S", f0: { kind: "Z" } } }),
    { kind: "Nil" }
  );
}
assertions();

function validations() {
  assert.deepEqual(
    list_drop({
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
    })({ kind: "S", f0: { kind: "S", f0: { kind: "Z" } } }),
    {
      kind: "Cons",
      f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
      f1: { kind: "Nil" },
    }
  );
  assert.deepEqual(
    list_drop({
      kind: "Cons",
      f0: { kind: "Z" },
      f1: {
        kind: "Cons",
        f0: { kind: "S", f0: { kind: "Z" } },
        f1: {
          kind: "Cons",
          f0: { kind: "S", f0: { kind: "Z" } },
          f1: { kind: "Nil" },
        },
      },
    })({ kind: "S", f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } } }),
    { kind: "Nil" }
  );
  assert.deepEqual(
    list_drop({
      kind: "Cons",
      f0: { kind: "Z" },
      f1: {
        kind: "Cons",
        f0: { kind: "S", f0: { kind: "Z" } },
        f1: {
          kind: "Cons",
          f0: { kind: "S", f0: { kind: "Z" } },
          f1: { kind: "Nil" },
        },
      },
    })({ kind: "S", f0: { kind: "S", f0: { kind: "Z" } } }),
    { kind: "Cons", f0: { kind: "S", f0: { kind: "Z" } }, f1: { kind: "Nil" } }
  );
}
validations();
