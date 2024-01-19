declare var require: any;
const assert = require("node:assert");
type nat = { kind: "Z" } | { kind: "S"; f0: nat };
type nlist = { kind: "Nil" } | { kind: "Cons"; f0: nat; f1: nlist };
type llist = { kind: "LNil" } | { kind: "LCons"; f0: nlist; f1: llist };

function append(l1: nlist): (l2: nlist) => nlist {
  return function (l2: nlist) {
    switch (l1.kind) {
      case "Nil": {
        return l2;
      }
      case "Cons": {
        let l1p = l1.f1;
        let x = l1.f0;
        return { kind: "Cons", f0: x, f1: append(l1p)(l2) };
      }
    }
  };
}

function concat(lol: llist): nlist {
  switch (lol.kind) {
    case "LNil": {
      return { kind: "Nil" };
    }
    case "LCons": {
      let ls = lol.f1;
      let l = lol.f0;
      return append(l)(concat(ls));
    }
  }
}

function assertions() {
  assert.deepEqual(concat({ kind: "LNil" }), { kind: "Nil" });
  assert.deepEqual(
    concat({
      kind: "LCons",
      f0: { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } },
      f1: { kind: "LNil" },
    }),
    { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } }
  );
  assert.deepEqual(
    concat({
      kind: "LCons",
      f0: { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } },
      f1: {
        kind: "LCons",
        f0: { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } },
        f1: { kind: "LNil" },
      },
    }),
    {
      kind: "Cons",
      f0: { kind: "Z" },
      f1: { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } },
    }
  );
  assert.deepEqual(
    concat({
      kind: "LCons",
      f0: {
        kind: "Cons",
        f0: { kind: "S", f0: { kind: "Z" } },
        f1: { kind: "Nil" },
      },
      f1: { kind: "LNil" },
    }),
    { kind: "Cons", f0: { kind: "S", f0: { kind: "Z" } }, f1: { kind: "Nil" } }
  );
  assert.deepEqual(
    concat({
      kind: "LCons",
      f0: {
        kind: "Cons",
        f0: { kind: "S", f0: { kind: "Z" } },
        f1: { kind: "Nil" },
      },
      f1: {
        kind: "LCons",
        f0: {
          kind: "Cons",
          f0: { kind: "S", f0: { kind: "Z" } },
          f1: { kind: "Nil" },
        },
        f1: { kind: "LNil" },
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
}
assertions();

function validations() {
  assert.deepEqual(
    concat({
      kind: "LCons",
      f0: { kind: "Nil" },
      f1: {
        kind: "LCons",
        f0: { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } },
        f1: {
          kind: "LCons",
          f0: { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } },
          f1: { kind: "LNil" },
        },
      },
    }),
    {
      kind: "Cons",
      f0: { kind: "Z" },
      f1: { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } },
    }
  );
  assert.deepEqual(
    concat({
      kind: "LCons",
      f0: {
        kind: "Cons",
        f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
        f1: { kind: "Nil" },
      },
      f1: { kind: "LNil" },
    }),
    {
      kind: "Cons",
      f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
      f1: { kind: "Nil" },
    }
  );
  assert.deepEqual(
    concat({
      kind: "LCons",
      f0: {
        kind: "Cons",
        f0: { kind: "S", f0: { kind: "Z" } },
        f1: { kind: "Nil" },
      },
      f1: {
        kind: "LCons",
        f0: {
          kind: "Cons",
          f0: {
            kind: "S",
            f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
          },
          f1: { kind: "Nil" },
        },
        f1: { kind: "LNil" },
      },
    }),
    {
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
    }
  );
}
validations();
