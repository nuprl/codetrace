declare var require: any;
const assert = require("node:assert");
type nat = { kind: "Z" } | { kind: "S"; f0: nat };
type nlist = { kind: "Nil" } | { kind: "Cons"; f0: nat; f1: nlist };

function list_tl(l: nlist): nlist {
  switch (l.kind) {
    case "Nil": {
      return { kind: "Nil" };
    }
    case "Cons": {
      let xs = l.f1;
      let x = l.f0;
      return xs;
    }
  }
}

function assertions() {
  assert.deepEqual(list_tl({ kind: "Nil" }), { kind: "Nil" });
  assert.deepEqual(
    list_tl({
      kind: "Cons",
      f0: { kind: "Z" },
      f1: { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } },
    }),
    { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } }
  );
  assert.deepEqual(
    list_tl({
      kind: "Cons",
      f0: { kind: "Z" },
      f1: {
        kind: "Cons",
        f0: { kind: "S", f0: { kind: "Z" } },
        f1: { kind: "Nil" },
      },
    }),
    { kind: "Cons", f0: { kind: "S", f0: { kind: "Z" } }, f1: { kind: "Nil" } }
  );
}
assertions();

function validations() {
  assert.deepEqual(list_tl({ kind: "Nil" }), { kind: "Nil" });
  assert.deepEqual(
    list_tl({
      kind: "Cons",
      f0: { kind: "S", f0: { kind: "Z" } },
      f1: { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } },
    }),
    { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } }
  );
  assert.deepEqual(
    list_tl({
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
      f0: { kind: "S", f0: { kind: "Z" } },
      f1: { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } },
    }
  );
}
validations();
