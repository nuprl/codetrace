declare var require: any;
const assert = require("node:assert");
type nat = { kind: "Z" } | { kind: "S"; f0: nat };
type nlist = { kind: "Nil" } | { kind: "Cons"; f0: nat; f1: nlist };

function list_append(l1: nlist, l2: nlist): nlist {
  switch (l1.kind) {
    case "Nil": {
      return l2;
    }
    case "Cons": {
      let xs = l1.f1;
      let x = l1.f0;
      return { kind: "Cons", f0: x, f1: list_append(xs, l2) };
    }
  }
}

function assertions() {
  assert.deepEqual(list_append({ kind: "Nil" }, { kind: "Nil" }), {
    kind: "Nil",
  });
  assert.deepEqual(
    list_append(
      {
        kind: "Cons",
        f0: { kind: "S", f0: { kind: "Z" } },
        f1: { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } },
      },
      { kind: "Nil" }
    ),
    {
      kind: "Cons",
      f0: { kind: "S", f0: { kind: "Z" } },
      f1: { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } },
    }
  );
  assert.deepEqual(
    list_append(
      { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } },
      { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } }
    ),
    {
      kind: "Cons",
      f0: { kind: "Z" },
      f1: { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } },
    }
  );
  assert.deepEqual(
    list_append(
      {
        kind: "Cons",
        f0: { kind: "S", f0: { kind: "Z" } },
        f1: { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } },
      },
      { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } }
    ),
    {
      kind: "Cons",
      f0: { kind: "S", f0: { kind: "Z" } },
      f1: {
        kind: "Cons",
        f0: { kind: "Z" },
        f1: { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } },
      },
    }
  );
}
assertions();

function validations() {
  assert.deepEqual(
    list_append(
      { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } },
      { kind: "Nil" }
    ),
    { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } }
  );
  assert.deepEqual(
    list_append(
      {
        kind: "Cons",
        f0: { kind: "S", f0: { kind: "Z" } },
        f1: { kind: "Nil" },
      },
      {
        kind: "Cons",
        f0: { kind: "S", f0: { kind: "Z" } },
        f1: { kind: "Nil" },
      }
    ),
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
  assert.deepEqual(
    list_append(
      {
        kind: "Cons",
        f0: { kind: "Z" },
        f1: { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } },
      },
      { kind: "Nil" }
    ),
    {
      kind: "Cons",
      f0: { kind: "Z" },
      f1: { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } },
    }
  );
}
validations();
