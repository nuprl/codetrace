declare var require: any;
const assert = require("node:assert");
type nat = { kind: "Z" } | { kind: "S"; f0: nat };
type nlist = { kind: "Nil" } | { kind: "Cons"; f0: nat; f1: nlist };

function list_nth(l: nlist, n: nat): nat {
  switch (l.kind) {
    case "Nil": {
      return { kind: "Z" };
    }
    case "Cons": {
      let xs = l.f1;
      let x = l.f0;
      switch (n.kind) {
        case "Z": {
          return x;
        }
        case "S": {
          let np = n.f0;
          return list_nth(xs, np);
        }
      }
    }
  }
}

function assertions() {
  assert.deepEqual(list_nth({ kind: "Nil" }, { kind: "Z" }), { kind: "Z" });
  assert.deepEqual(
    list_nth(
      {
        kind: "Cons",
        f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
        f1: { kind: "Nil" },
      },
      { kind: "S", f0: { kind: "Z" } }
    ),
    { kind: "Z" }
  );
  assert.deepEqual(
    list_nth(
      {
        kind: "Cons",
        f0: { kind: "S", f0: { kind: "Z" } },
        f1: {
          kind: "Cons",
          f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
          f1: { kind: "Nil" },
        },
      },
      { kind: "Z" }
    ),
    { kind: "S", f0: { kind: "Z" } }
  );
  assert.deepEqual(
    list_nth(
      {
        kind: "Cons",
        f0: { kind: "S", f0: { kind: "Z" } },
        f1: {
          kind: "Cons",
          f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
          f1: { kind: "Nil" },
        },
      },
      { kind: "S", f0: { kind: "Z" } }
    ),
    { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } }
  );
  assert.deepEqual(
    list_nth(
      {
        kind: "Cons",
        f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
        f1: {
          kind: "Cons",
          f0: { kind: "S", f0: { kind: "Z" } },
          f1: { kind: "Nil" },
        },
      },
      { kind: "Z" }
    ),
    { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } }
  );
  assert.deepEqual(
    list_nth(
      {
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
      },
      { kind: "Z" }
    ),
    { kind: "S", f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } } }
  );
  assert.deepEqual(
    list_nth(
      {
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
        },
      },
      { kind: "S", f0: { kind: "Z" } }
    ),
    { kind: "S", f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } } }
  );
}
assertions();

function validations() {
  assert.deepEqual(
    list_nth(
      {
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
      },
      { kind: "Z" }
    ),
    {
      kind: "S",
      f0: {
        kind: "S",
        f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
      },
    }
  );
  assert.deepEqual(
    list_nth(
      {
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
      },
      { kind: "S", f0: { kind: "Z" } }
    ),
    { kind: "S", f0: { kind: "Z" } }
  );
  assert.deepEqual(
    list_nth(
      {
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
      },
      { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } }
    ),
    { kind: "S", f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } } }
  );
  assert.deepEqual(
    list_nth(
      {
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
      },
      { kind: "S", f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } } }
    ),
    { kind: "Z" }
  );
}
validations();
