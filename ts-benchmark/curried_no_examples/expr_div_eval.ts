declare var require: any;
const assert = require("node:assert");
type nat = { kind: "Z" } | { kind: "S"; f0: nat };
type exp =
  | { kind: "Int"; f0: nat }
  | { kind: "Add"; f0: exp; f1: exp }
  | { kind: "Sub"; f0: exp; f1: exp }
  | { kind: "Mul"; f0: exp; f1: exp }
  | { kind: "Div"; f0: exp; f1: exp };
type cmp = { kind: "LT" } | { kind: "EQ" } | { kind: "GT" };

function compare(x1: nat): (x2: nat) => cmp {
  return function (x2: nat) {
    switch (x1.kind) {
      case "Z": {
        switch (x2.kind) {
          case "Z": {
            return { kind: "EQ" };
          }
          case "S": {
            let n1 = x2.f0;
            return { kind: "LT" };
          }
        }
      }
      case "S": {
        let n1 = x1.f0;
        switch (x2.kind) {
          case "Z": {
            return { kind: "GT" };
          }
          case "S": {
            let n2 = x2.f0;
            return compare(n1)(n2);
          }
        }
      }
    }
  };
}
function add(n1: nat): (n2: nat) => nat {
  return function (n2: nat) {
    switch (n1.kind) {
      case "Z": {
        return n2;
      }
      case "S": {
        let n3 = n1.f0;
        return { kind: "S", f0: add(n3)(n2) };
      }
    }
  };
}
function sub(n1: nat): (n2: nat) => nat {
  return function (n2: nat) {
    switch (n1.kind) {
      case "Z": {
        return { kind: "Z" };
      }
      case "S": {
        let n3 = n1.f0;
        switch (n2.kind) {
          case "Z": {
            return n1;
          }
          case "S": {
            let n4 = n2.f0;
            return sub(n3)(n4);
          }
        }
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
function div(n1: nat): (n2: nat) => nat {
  return function (n2: nat) {
    switch (n2.kind) {
      case "Z": {
        return { kind: "Z" };
      }
      case "S": {
        let n4 = n2.f0;
        switch (n1.kind) {
          case "Z": {
            return { kind: "Z" };
          }
          case "S": {
            let n3 = n1.f0;
            switch (compare(n1)(n2).kind) {
              case "LT": {
                return { kind: "Z" };
              }
              case "EQ": {
                return { kind: "S", f0: { kind: "Z" } };
              }
              case "GT": {
                return { kind: "S", f0: div(sub(n1)(n2))(n2) };
              }
            }
          }
        }
      }
    }
  };
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
      return add(expr_eval(e1))(expr_eval(e2));
    }
    case "Mul": {
      let e2 = e.f1;
      let e1 = e.f0;
      return mul(expr_eval(e1))(expr_eval(e2));
    }
    case "Sub": {
      let e2 = e.f1;
      let e1 = e.f0;
      return sub(expr_eval(e1))(expr_eval(e2));
    }
    case "Div": {
      let e2 = e.f1;
      let e1 = e.f0;
      return div(expr_eval(e1))(expr_eval(e2));
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
  assert.deepEqual(
    expr_eval({
      kind: "Sub",
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
          f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
        },
      },
    }),
    { kind: "S", f0: { kind: "Z" } }
  );
  assert.deepEqual(
    expr_eval({
      kind: "Sub",
      f0: {
        kind: "Int",
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
      f1: { kind: "Int", f0: { kind: "S", f0: { kind: "Z" } } },
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
    expr_eval({
      kind: "Div",
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
        f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
      },
    }),
    { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } }
  );
  assert.deepEqual(
    expr_eval({
      kind: "Div",
      f0: {
        kind: "Int",
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
        kind: "Int",
        f0: {
          kind: "S",
          f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
        },
      },
    }),
    { kind: "S", f0: { kind: "Z" } }
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
  assert.deepEqual(
    expr_eval({
      kind: "Div",
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
          f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
        },
      },
    }),
    { kind: "S", f0: { kind: "Z" } }
  );
}
validations();
