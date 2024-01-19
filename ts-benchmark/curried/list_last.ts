declare var require: any;
const assert = require("node:assert");
type nat = { kind: "Z" } | { kind: "S"; f0: nat };
type nlist = { kind: "Nil" } | { kind: "Cons"; f0: nat; f1: nlist };
type natopt = { kind: "None" } | { kind: "Some"; f0: nat };

function list_last(l: nlist): natopt {
  switch (l.kind) {
    case "Nil": {
      return { kind: "None" };
    }
    case "Cons": {
      let xs = l.f1;
      let x = l.f0;
      switch (xs.kind) {
        case "Nil": {
          return { kind: "Some", f0: x };
        }
        case "Cons": {
          let xsp = xs.f1;
          let xp = xs.f0;
          return list_last(xs);
        }
      }
    }
  }
}

function assertions() {
  assert.deepEqual(list_last({ kind: "Nil" }), { kind: "None" });
  assert.deepEqual(
    list_last({
      kind: "Cons",
      f0: { kind: "S", f0: { kind: "Z" } },
      f1: { kind: "Nil" },
    }),
    { kind: "Some", f0: { kind: "S", f0: { kind: "Z" } } }
  );
  assert.deepEqual(
    list_last({
      kind: "Cons",
      f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
      f1: { kind: "Nil" },
    }),
    { kind: "Some", f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } } }
  );
  assert.deepEqual(
    list_last({
      kind: "Cons",
      f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
      f1: {
        kind: "Cons",
        f0: { kind: "S", f0: { kind: "Z" } },
        f1: { kind: "Nil" },
      },
    }),
    { kind: "Some", f0: { kind: "S", f0: { kind: "Z" } } }
  );
  assert.deepEqual(
    list_last({
      kind: "Cons",
      f0: { kind: "S", f0: { kind: "Z" } },
      f1: {
        kind: "Cons",
        f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
        f1: { kind: "Nil" },
      },
    }),
    { kind: "Some", f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } } }
  );
  assert.deepEqual(
    list_last({
      kind: "Cons",
      f0: {
        kind: "S",
        f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
      },
      f1: {
        kind: "Cons",
        f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
        f1: {
          kind: "Cons",
          f0: { kind: "S", f0: { kind: "Z" } },
          f1: { kind: "Nil" },
        },
      },
    }),
    { kind: "Some", f0: { kind: "S", f0: { kind: "Z" } } }
  );
}
assertions();

function validations() {
  assert.deepEqual(
    list_last({
      kind: "Cons",
      f0: {
        kind: "S",
        f0: {
          kind: "S",
          f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
        },
      },
      f1: {
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
      },
    }),
    {
      kind: "Some",
      f0: {
        kind: "S",
        f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
      },
    }
  );
  assert.deepEqual(
    list_last({
      kind: "Cons",
      f0: {
        kind: "S",
        f0: {
          kind: "S",
          f0: {
            kind: "S",
            f0: {
              kind: "S",
              f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
            },
          },
        },
      },
      f1: {
        kind: "Cons",
        f0: { kind: "S", f0: { kind: "Z" } },
        f1: {
          kind: "Cons",
          f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
          f1: { kind: "Nil" },
        },
      },
    }),
    { kind: "Some", f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } } }
  );
  assert.deepEqual(
    list_last({
      kind: "Cons",
      f0: { kind: "S", f0: { kind: "Z" } },
      f1: {
        kind: "Cons",
        f0: { kind: "S", f0: { kind: "Z" } },
        f1: {
          kind: "Cons",
          f0: { kind: "S", f0: { kind: "Z" } },
          f1: {
            kind: "Cons",
            f0: { kind: "S", f0: { kind: "Z" } },
            f1: { kind: "Nil" },
          },
        },
      },
    }),
    { kind: "Some", f0: { kind: "S", f0: { kind: "Z" } } }
  );
}
validations();
