declare var require: any;
const assert = require("node:assert");
type nat = { kind: "Z" } | { kind: "S"; f0: nat };
type bbool = { kind: "True" } | { kind: "False" };
type blist = { kind: "Nil" } | { kind: "Cons"; f0: bbool; f1: blist };

function list_even_parity(l: blist): bbool {
  switch (l.kind) {
    case "Nil": {
      return { kind: "True" };
    }
    case "Cons": {
      let xs = l.f1;
      let x = l.f0;
      let xs_parity: bbool = list_even_parity(xs);
      switch (x.kind) {
        case "True": {
          switch (xs_parity.kind) {
            case "True": {
              return { kind: "False" };
            }
            case "False": {
              return { kind: "True" };
            }
          }
        }
        case "False": {
          return xs_parity;
        }
      }
    }
  }
}

function assertions() {
  assert.deepEqual(list_even_parity({ kind: "Nil" }), { kind: "True" });
  assert.deepEqual(
    list_even_parity({
      kind: "Cons",
      f0: { kind: "False" },
      f1: { kind: "Nil" },
    }),
    { kind: "True" }
  );
  assert.deepEqual(
    list_even_parity({
      kind: "Cons",
      f0: { kind: "True" },
      f1: { kind: "Nil" },
    }),
    { kind: "False" }
  );
  assert.deepEqual(
    list_even_parity({
      kind: "Cons",
      f0: { kind: "False" },
      f1: { kind: "Cons", f0: { kind: "False" }, f1: { kind: "Nil" } },
    }),
    { kind: "True" }
  );
  assert.deepEqual(
    list_even_parity({
      kind: "Cons",
      f0: { kind: "False" },
      f1: { kind: "Cons", f0: { kind: "True" }, f1: { kind: "Nil" } },
    }),
    { kind: "False" }
  );
  assert.deepEqual(
    list_even_parity({
      kind: "Cons",
      f0: { kind: "True" },
      f1: { kind: "Cons", f0: { kind: "False" }, f1: { kind: "Nil" } },
    }),
    { kind: "False" }
  );
  assert.deepEqual(
    list_even_parity({
      kind: "Cons",
      f0: { kind: "True" },
      f1: { kind: "Cons", f0: { kind: "True" }, f1: { kind: "Nil" } },
    }),
    { kind: "True" }
  );
}
assertions();

function validations() {
  assert.deepEqual(
    list_even_parity({
      kind: "Cons",
      f0: { kind: "False" },
      f1: {
        kind: "Cons",
        f0: { kind: "False" },
        f1: { kind: "Cons", f0: { kind: "False" }, f1: { kind: "Nil" } },
      },
    }),
    { kind: "True" }
  );
  assert.deepEqual(
    list_even_parity({
      kind: "Cons",
      f0: { kind: "False" },
      f1: {
        kind: "Cons",
        f0: { kind: "True" },
        f1: { kind: "Cons", f0: { kind: "False" }, f1: { kind: "Nil" } },
      },
    }),
    { kind: "False" }
  );
  assert.deepEqual(
    list_even_parity({
      kind: "Cons",
      f0: { kind: "True" },
      f1: {
        kind: "Cons",
        f0: { kind: "False" },
        f1: { kind: "Cons", f0: { kind: "True" }, f1: { kind: "Nil" } },
      },
    }),
    { kind: "True" }
  );
  assert.deepEqual(
    list_even_parity({
      kind: "Cons",
      f0: { kind: "True" },
      f1: {
        kind: "Cons",
        f0: { kind: "True" },
        f1: { kind: "Cons", f0: { kind: "True" }, f1: { kind: "Nil" } },
      },
    }),
    { kind: "False" }
  );
}
validations();
