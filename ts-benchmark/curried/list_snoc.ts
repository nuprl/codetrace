declare var require: any;
const assert = require("node:assert");
type nat = { kind: "Z" } | { kind: "S"; f0: nat };
type nlist = { kind: "Nil" } | { kind: "Cons"; f0: nat; f1: nlist };

function snoc(l: nlist): (n: nat) => nlist {
  return function (n: nat) {
    switch (l.kind) {
      case "Nil": {
        return { kind: "Cons", f0: n, f1: { kind: "Nil" } };
      }
      case "Cons": {
        let xs = l.f1;
        let x = l.f0;
        return { kind: "Cons", f0: x, f1: snoc(xs)(n) };
      }
    }
  };
}

function assertions() {
  assert.deepEqual(snoc({ kind: "Nil" })({ kind: "Z" }), {
    kind: "Cons",
    f0: { kind: "Z" },
    f1: { kind: "Nil" },
  });
  assert.deepEqual(snoc({ kind: "Nil" })({ kind: "S", f0: { kind: "Z" } }), {
    kind: "Cons",
    f0: { kind: "S", f0: { kind: "Z" } },
    f1: { kind: "Nil" },
  });
  assert.deepEqual(
    snoc({
      kind: "Cons",
      f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
      f1: {
        kind: "Cons",
        f0: { kind: "S", f0: { kind: "Z" } },
        f1: { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } },
      },
    })({ kind: "S", f0: { kind: "Z" } }),
    {
      kind: "Cons",
      f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
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
    }
  );
}
assertions();

function validations() {
  assert.deepEqual(
    snoc({
      kind: "Cons",
      f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
      f1: {
        kind: "Cons",
        f0: { kind: "S", f0: { kind: "Z" } },
        f1: { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } },
      },
    })({ kind: "Z" }),
    {
      kind: "Cons",
      f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
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
  assert.deepEqual(
    snoc({
      kind: "Cons",
      f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
      f1: {
        kind: "Cons",
        f0: { kind: "Z" },
        f1: { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } },
      },
    })({ kind: "S", f0: { kind: "S", f0: { kind: "Z" } } }),
    {
      kind: "Cons",
      f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
      f1: {
        kind: "Cons",
        f0: { kind: "Z" },
        f1: {
          kind: "Cons",
          f0: { kind: "Z" },
          f1: {
            kind: "Cons",
            f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
            f1: { kind: "Nil" },
          },
        },
      },
    }
  );
  assert.deepEqual(
    snoc({
      kind: "Cons",
      f0: { kind: "S", f0: { kind: "Z" } },
      f1: { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } },
    })({ kind: "S", f0: { kind: "S", f0: { kind: "Z" } } }),
    {
      kind: "Cons",
      f0: { kind: "S", f0: { kind: "Z" } },
      f1: {
        kind: "Cons",
        f0: { kind: "Z" },
        f1: {
          kind: "Cons",
          f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
          f1: { kind: "Nil" },
        },
      },
    }
  );
  assert.deepEqual(
    snoc({ kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } })({
      kind: "Z",
    }),
    {
      kind: "Cons",
      f0: { kind: "Z" },
      f1: { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } },
    }
  );
}
validations();
