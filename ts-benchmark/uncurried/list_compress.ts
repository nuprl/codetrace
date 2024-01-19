declare var require: any;
const assert = require("node:assert");
type nat = { kind: "Z" } | { kind: "S"; f0: nat };
type nlist = { kind: "Nil" } | { kind: "Cons"; f0: nat; f1: nlist };
type cmp = { kind: "LT" } | { kind: "EQ" } | { kind: "GT" };

function compare(n1: nat, n2: nat): cmp {
  switch (n1.kind) {
    case "Z": {
      switch (n2.kind) {
        case "Z": {
          return { kind: "EQ" };
        }
        case "S": {
          let m = n2.f0;
          return { kind: "LT" };
        }
      }
    }
    case "S": {
      let m1 = n1.f0;
      switch (n2.kind) {
        case "Z": {
          return { kind: "GT" };
        }
        case "S": {
          let m2 = n2.f0;
          return compare(m1, m2);
        }
      }
    }
  }
}

function list_compress(lox: nlist): nlist {
  switch (lox.kind) {
    case "Nil": {
      return { kind: "Nil" };
    }
    case "Cons": {
      let xs = lox.f1;
      let x = lox.f0;
      switch (xs.kind) {
        case "Nil": {
          return lox;
        }
        case "Cons": {
          let ys = xs.f1;
          let y = xs.f0;
          switch (compare(x, y).kind) {
            case "EQ": {
              return list_compress(xs);
            }
            case "LT": {
              return { kind: "Cons", f0: x, f1: list_compress(xs) };
            }
            case "GT": {
              return { kind: "Cons", f0: x, f1: list_compress(xs) };
            }
          }
        }
      }
    }
  }
}

function assertions() {
  assert.deepEqual(list_compress({ kind: "Nil" }), { kind: "Nil" });
  assert.deepEqual(
    list_compress({ kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } }),
    { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } }
  );
  assert.deepEqual(
    list_compress({
      kind: "Cons",
      f0: { kind: "S", f0: { kind: "Z" } },
      f1: { kind: "Nil" },
    }),
    { kind: "Cons", f0: { kind: "S", f0: { kind: "Z" } }, f1: { kind: "Nil" } }
  );
  assert.deepEqual(
    list_compress({
      kind: "Cons",
      f0: { kind: "Z" },
      f1: { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } },
    }),
    { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } }
  );
  assert.deepEqual(
    list_compress({
      kind: "Cons",
      f0: { kind: "S", f0: { kind: "Z" } },
      f1: {
        kind: "Cons",
        f0: { kind: "S", f0: { kind: "Z" } },
        f1: { kind: "Nil" },
      },
    }),
    { kind: "Cons", f0: { kind: "S", f0: { kind: "Z" } }, f1: { kind: "Nil" } }
  );
  assert.deepEqual(
    list_compress({
      kind: "Cons",
      f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
      f1: { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } },
    }),
    {
      kind: "Cons",
      f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
      f1: { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } },
    }
  );
  assert.deepEqual(
    list_compress({
      kind: "Cons",
      f0: { kind: "S", f0: { kind: "Z" } },
      f1: {
        kind: "Cons",
        f0: { kind: "Z" },
        f1: { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } },
      },
    }),
    {
      kind: "Cons",
      f0: { kind: "S", f0: { kind: "Z" } },
      f1: { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } },
    }
  );
  assert.deepEqual(
    list_compress({
      kind: "Cons",
      f0: { kind: "Z" },
      f1: {
        kind: "Cons",
        f0: { kind: "S", f0: { kind: "Z" } },
        f1: {
          kind: "Cons",
          f0: { kind: "S", f0: { kind: "Z" } },
          f1: { kind: "Nil" },
        },
      },
    }),
    {
      kind: "Cons",
      f0: { kind: "Z" },
      f1: {
        kind: "Cons",
        f0: { kind: "S", f0: { kind: "Z" } },
        f1: { kind: "Nil" },
      },
    }
  );
  assert.deepEqual(
    list_compress({
      kind: "Cons",
      f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
      f1: {
        kind: "Cons",
        f0: { kind: "S", f0: { kind: "Z" } },
        f1: {
          kind: "Cons",
          f0: { kind: "Z" },
          f1: { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } },
        },
      },
    }),
    {
      kind: "Cons",
      f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
      f1: {
        kind: "Cons",
        f0: { kind: "S", f0: { kind: "Z" } },
        f1: { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } },
      },
    }
  );
  assert.deepEqual(
    list_compress({
      kind: "Cons",
      f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
      f1: {
        kind: "Cons",
        f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
        f1: {
          kind: "Cons",
          f0: { kind: "S", f0: { kind: "Z" } },
          f1: {
            kind: "Cons",
            f0: { kind: "Z" },
            f1: { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } },
          },
        },
      },
    }),
    {
      kind: "Cons",
      f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
      f1: {
        kind: "Cons",
        f0: { kind: "S", f0: { kind: "Z" } },
        f1: { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } },
      },
    }
  );
  assert.deepEqual(
    list_compress({
      kind: "Cons",
      f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
      f1: {
        kind: "Cons",
        f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
        f1: { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } },
      },
    }),
    {
      kind: "Cons",
      f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
      f1: { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } },
    }
  );
  assert.deepEqual(
    list_compress({
      kind: "Cons",
      f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
      f1: {
        kind: "Cons",
        f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
        f1: {
          kind: "Cons",
          f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
          f1: { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } },
        },
      },
    }),
    {
      kind: "Cons",
      f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
      f1: { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } },
    }
  );
  assert.deepEqual(
    list_compress({
      kind: "Cons",
      f0: { kind: "S", f0: { kind: "Z" } },
      f1: {
        kind: "Cons",
        f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
        f1: {
          kind: "Cons",
          f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
          f1: {
            kind: "Cons",
            f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
            f1: { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } },
          },
        },
      },
    }),
    {
      kind: "Cons",
      f0: { kind: "S", f0: { kind: "Z" } },
      f1: {
        kind: "Cons",
        f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
        f1: { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } },
      },
    }
  );
}
assertions();

function validations() {
  assert.deepEqual(
    list_compress({
      kind: "Cons",
      f0: { kind: "Z" },
      f1: {
        kind: "Cons",
        f0: { kind: "Z" },
        f1: { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } },
      },
    }),
    { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } }
  );
  assert.deepEqual(
    list_compress({
      kind: "Cons",
      f0: { kind: "Z" },
      f1: {
        kind: "Cons",
        f0: { kind: "Z" },
        f1: {
          kind: "Cons",
          f0: { kind: "Z" },
          f1: { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } },
        },
      },
    }),
    { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } }
  );
  assert.deepEqual(
    list_compress({
      kind: "Cons",
      f0: { kind: "S", f0: { kind: "Z" } },
      f1: {
        kind: "Cons",
        f0: { kind: "S", f0: { kind: "Z" } },
        f1: {
          kind: "Cons",
          f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
          f1: {
            kind: "Cons",
            f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
            f1: {
              kind: "Cons",
              f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
              f1: { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } },
            },
          },
        },
      },
    }),
    {
      kind: "Cons",
      f0: { kind: "S", f0: { kind: "Z" } },
      f1: {
        kind: "Cons",
        f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
        f1: { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } },
      },
    }
  );
}
validations();
