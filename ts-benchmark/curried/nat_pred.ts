declare var require: any;
const assert = require("node:assert");
type nat = { kind: "Z" } | { kind: "S"; f0: nat };

function nat_pred(n: nat): nat {
  switch (n.kind) {
    case "Z": {
      return { kind: "Z" };
    }
    case "S": {
      let n1 = n.f0;
      return n1;
    }
  }
}

function assertions() {
  assert.deepEqual(nat_pred({ kind: "Z" }), { kind: "Z" });
  assert.deepEqual(
    nat_pred({ kind: "S", f0: { kind: "S", f0: { kind: "Z" } } }),
    { kind: "S", f0: { kind: "Z" } }
  );
}
assertions();

function validations() {
  assert.deepEqual(nat_pred({ kind: "Z" }), { kind: "Z" });
  assert.deepEqual(
    nat_pred({
      kind: "S",
      f0: {
        kind: "S",
        f0: {
          kind: "S",
          f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
        },
      },
    }),
    {
      kind: "S",
      f0: {
        kind: "S",
        f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
      },
    }
  );
  assert.deepEqual(
    nat_pred({
      kind: "S",
      f0: {
        kind: "S",
        f0: {
          kind: "S",
          f0: {
            kind: "S",
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
          },
        },
      },
    }),
    {
      kind: "S",
      f0: {
        kind: "S",
        f0: {
          kind: "S",
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
        },
      },
    }
  );
  assert.deepEqual(nat_pred({ kind: "S", f0: { kind: "Z" } }), { kind: "Z" });
}
validations();
