declare var require: any;
const assert = require("node:assert");
type nat = { kind: "Z" } | { kind: "S"; f0: nat };

function add(x: nat): (y: nat) => nat {
  return function (y: nat) {
    switch (x.kind) {
      case "Z": {
        return y;
      }
      case "S": {
        let xp = x.f0;
        return { kind: "S", f0: add(xp)(y) };
      }
    }
  };
}

function mul(n1: nat): (n2: nat) => nat {
  return function (n2: nat) {
    switch (n1.kind) {
      case "Z": {
        return { kind: "Z" };
      }
      case "S": {
        let n3 = n1.f0;
        return add(n2)(mul(n3)(n2));
      }
    }
  };
}

function assertions() {
  assert.deepEqual(mul({ kind: "Z" })({ kind: "Z" }), { kind: "Z" });
  assert.deepEqual(mul({ kind: "S", f0: { kind: "Z" } })({ kind: "Z" }), {
    kind: "Z",
  });
  assert.deepEqual(
    mul({ kind: "S", f0: { kind: "Z" } })({
      kind: "S",
      f0: { kind: "S", f0: { kind: "Z" } },
    }),
    { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } }
  );
  assert.deepEqual(
    mul({ kind: "S", f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } } })(
      { kind: "S", f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } } }
    ),
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
  assert.deepEqual(
    mul({ kind: "S", f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } } })(
      { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } }
    ),
    {
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
    }
  );
  assert.deepEqual(
    mul({ kind: "S", f0: { kind: "S", f0: { kind: "Z" } } })({
      kind: "S",
      f0: { kind: "S", f0: { kind: "Z" } },
    }),
    {
      kind: "S",
      f0: {
        kind: "S",
        f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
      },
    }
  );
}
assertions();

function validations() {
  assert.deepEqual(
    mul({ kind: "S", f0: { kind: "S", f0: { kind: "Z" } } })({
      kind: "S",
      f0: { kind: "S", f0: { kind: "Z" } },
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
    mul({ kind: "Z" })({ kind: "S", f0: { kind: "S", f0: { kind: "Z" } } }),
    { kind: "Z" }
  );
  assert.deepEqual(
    mul({ kind: "S", f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } } })(
      {
        kind: "S",
        f0: {
          kind: "S",
          f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
        },
      }
    ),
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
        },
      },
    }
  );
  assert.deepEqual(
    mul({ kind: "S", f0: { kind: "Z" } })({
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
        f0: {
          kind: "S",
          f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
        },
      },
    }
  );
}
validations();
