declare var require: any;
const assert = require("node:assert");
type nat = { kind: "Z" } | { kind: "S"; f0: nat };
type nlist = { kind: "Nil" } | { kind: "Cons"; f0: nat; f1: nlist };

function fold(
  l: nlist,
  f: (__x5: nlist, __x6: nat) => nlist,
  acc: nlist
): nlist {
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
function snoc(l: nlist, n: nat): nlist {
  switch (l.kind) {
    case "Nil": {
      return { kind: "Cons", f0: n, f1: { kind: "Nil" } };
    }
    case "Cons": {
      let xs = l.f1;
      let x = l.f0;
      return { kind: "Cons", f0: x, f1: snoc(xs, n) };
    }
  }
}

function list_rev_fold(l: nlist): nlist {
  return fold(
    l,
    function (l: nlist, n: nat) {
      return { kind: "Cons", f0: n, f1: l };
    },
    { kind: "Nil" }
  );
}

function assertions() {
  assert.deepEqual(list_rev_fold({ kind: "Nil" }), { kind: "Nil" });
  assert.deepEqual(
    list_rev_fold({
      kind: "Cons",
      f0: { kind: "Z" },
      f1: {
        kind: "Cons",
        f0: { kind: "S", f0: { kind: "Z" } },
        f1: { kind: "Nil" },
      },
    }),
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
    list_rev_fold({
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
    list_rev_fold({
      kind: "Cons",
      f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
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
    list_rev_fold({
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
    }),
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
    list_rev_fold({
      kind: "Cons",
      f0: { kind: "S", f0: { kind: "Z" } },
      f1: { kind: "Nil" },
    }),
    { kind: "Cons", f0: { kind: "S", f0: { kind: "Z" } }, f1: { kind: "Nil" } }
  );
}
validations();
