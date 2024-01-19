declare var require: any;
const assert = require("node:assert");
type nat = { kind: "Z" } | { kind: "S"; f0: nat };
type exp =
  | { kind: "Int"; f0: nat }
  | { kind: "Add"; f0: exp; f1: exp }
  | { kind: "Mul"; f0: exp; f1: exp };

function add(n1: nat, n2: nat): nat {
  switch (n1.kind) {
    case "Z": {
      return n2;
    }
    case "S": {
      let n3 = n1.f0;
      return { kind: "S", f0: add(n3, n2) };
    }
  }
}
function mul(n1: nat, n2: nat): nat {
  switch (n1.kind) {
    case "Z": {
      return { kind: "Z" };
    }
    case "S": {
      let n3 = n1.f0;
      return add(n2, mul(n3, n2));
    }
  }
}

function expr_eval(e: exp): nat {
  switch (e.kind) {
    case "Int": {
      let n = e.f0;
      return n;
    }
    case "Add": {
      let e2 = e.f1;
      let e1 = e.f0;
      return add(expr_eval(e1), expr_eval(e2));
    }
    case "Mul": {
      let e2 = e.f1;
      let e1 = e.f0;
      return mul(expr_eval(e1), expr_eval(e2));
    }
  }
}

function assertions() {
  assert.deepEqual(
    expr_eval({ kind: "Int", f0: { kind: "S", f0: { kind: "Z" } } }),
    { kind: "S", f0: { kind: "Z" } }
  );
  assert.deepEqual(
    expr_eval({
      kind: "Add",
      f0: {
        kind: "Int",
        f0: {
          kind: "S",
          f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
        },
      },
      f1: {
        kind: "Int",
        f0: {
          kind: "S",
          f0: {
            kind: "S",
            f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
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
              f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
            },
          },
        },
      },
    }
  );
  assert.deepEqual(
    expr_eval({
      kind: "Mul",
      f0: {
        kind: "Int",
        f0: {
          kind: "S",
          f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
        },
      },
      f1: {
        kind: "Int",
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
    expr_eval({
      kind: "Mul",
      f0: {
        kind: "Int",
        f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
      },
      f1: {
        kind: "Int",
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
          f0: {
            kind: "S",
            f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
          },
        },
      },
    }
  );
}
assertions();

function validations() {
  assert.deepEqual(
    expr_eval({
      kind: "Add",
      f0: {
        kind: "Int",
        f0: {
          kind: "S",
          f0: {
            kind: "S",
            f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
          },
        },
      },
      f1: {
        kind: "Int",
        f0: {
          kind: "S",
          f0: {
            kind: "S",
            f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
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
                f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
              },
            },
          },
        },
      },
    }
  );
  assert.deepEqual(
    expr_eval({
      kind: "Mul",
      f0: {
        kind: "Int",
        f0: {
          kind: "S",
          f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
        },
      },
      f1: {
        kind: "Int",
        f0: {
          kind: "S",
          f0: {
            kind: "S",
            f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
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
}
validations();
