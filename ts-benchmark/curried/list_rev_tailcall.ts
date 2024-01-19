declare var require: any;
const assert = require("node:assert");
type nat = { kind: "Z" } | { kind: "S"; f0: nat };
type nlist = { kind: "Nil" } | { kind: "Cons"; f0: nat; f1: nlist };

function list_rev_tailcall(rem: nlist): (acc: nlist) => nlist {
  return function (acc: nlist) {
    switch (rem.kind) {
      case "Nil": {
        return acc;
      }
      case "Cons": {
        let xs = rem.f1;
        let x = rem.f0;
        return list_rev_tailcall(xs)({ kind: "Cons", f0: x, f1: acc });
      }
    }
  };
}

function assertions() {
  assert.deepEqual(list_rev_tailcall({ kind: "Nil" })({ kind: "Nil" }), {
    kind: "Nil",
  });
  assert.deepEqual(
    list_rev_tailcall({ kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } })(
      { kind: "Nil" }
    ),
    { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } }
  );
  assert.deepEqual(
    list_rev_tailcall({
      kind: "Cons",
      f0: { kind: "Z" },
      f1: {
        kind: "Cons",
        f0: { kind: "S", f0: { kind: "Z" } },
        f1: { kind: "Nil" },
      },
    })({ kind: "Nil" }),
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
    list_rev_tailcall({
      kind: "Cons",
      f0: { kind: "S", f0: { kind: "Z" } },
      f1: { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } },
    })({ kind: "Nil" }),
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
    list_rev_tailcall({
      kind: "Cons",
      f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
      f1: {
        kind: "Cons",
        f0: { kind: "S", f0: { kind: "Z" } },
        f1: { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } },
      },
    })({ kind: "Nil" }),
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
    list_rev_tailcall({
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
            f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
            f1: { kind: "Nil" },
          },
        },
      },
    })({ kind: "Nil" }),
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
    list_rev_tailcall({
      kind: "Cons",
      f0: { kind: "S", f0: { kind: "Z" } },
      f1: { kind: "Nil" },
    })({ kind: "Nil" }),
    { kind: "Cons", f0: { kind: "S", f0: { kind: "Z" } }, f1: { kind: "Nil" } }
  );
}
validations();
