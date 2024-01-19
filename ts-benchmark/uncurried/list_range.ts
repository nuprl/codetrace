declare var require: any;
const assert = require("node:assert");
type nat = { kind: "Z" } | { kind: "S"; f0: nat };
type nlist = { kind: "Nil" } | { kind: "Cons"; f0: nat; f1: nlist };
type cmp = { kind: "LT" } | { kind: "EQ" } | { kind: "GT" };

function compare(x1: nat, x2: nat): cmp {
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
          return compare(n1, n2);
        }
      }
    }
  }
}

function list_range(start: nat, fin: nat): nlist {
  switch (compare(start, fin).kind) {
    case "LT": {
      return { kind: "Cons", f0: start, f1: { kind: "Nil" } };
    }
    case "EQ": {
      return { kind: "Cons", f0: start, f1: { kind: "Nil" } };
    }
    case "GT": {
      switch (start.kind) {
        case "Z": {
          return { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } };
        }
        case "S": {
          let n = start.f0;
          return { kind: "Cons", f0: start, f1: list_range(n, fin) };
        }
      }
    }
  }
}

function assertions() {
  assert.deepEqual(list_range({ kind: "Z" }, { kind: "Z" }), {
    kind: "Cons",
    f0: { kind: "Z" },
    f1: { kind: "Nil" },
  });
  assert.deepEqual(
    list_range({ kind: "Z" }, { kind: "S", f0: { kind: "Z" } }),
    { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } }
  );
  assert.deepEqual(
    list_range(
      { kind: "S", f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } } },
      { kind: "S", f0: { kind: "Z" } }
    ),
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
    }
  );
  assert.deepEqual(
    list_range(
      {
        kind: "S",
        f0: {
          kind: "S",
          f0: {
            kind: "S",
            f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
          },
        },
      },
      { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } }
    ),
    {
      kind: "Cons",
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
      f1: {
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
            f1: { kind: "Nil" },
          },
        },
      },
    }
  );
  assert.deepEqual(
    list_range(
      { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
      { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } }
    ),
    {
      kind: "Cons",
      f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
      f1: { kind: "Nil" },
    }
  );
  assert.deepEqual(
    list_range(
      {
        kind: "S",
        f0: {
          kind: "S",
          f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
        },
      },
      { kind: "S", f0: { kind: "Z" } }
    ),
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
    }
  );
  assert.deepEqual(
    list_range({ kind: "S", f0: { kind: "Z" } }, { kind: "Z" }),
    {
      kind: "Cons",
      f0: { kind: "S", f0: { kind: "Z" } },
      f1: { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } },
    }
  );
}
assertions();

function validations() {
  assert.deepEqual(
    list_range(
      { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
      { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } }
    ),
    {
      kind: "Cons",
      f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
      f1: { kind: "Nil" },
    }
  );
  assert.deepEqual(
    list_range(
      { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
      { kind: "S", f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } } }
    ),
    {
      kind: "Cons",
      f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
      f1: { kind: "Nil" },
    }
  );
  assert.deepEqual(
    list_range(
      {
        kind: "S",
        f0: {
          kind: "S",
          f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
        },
      },
      { kind: "S", f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } } }
    ),
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
        f1: { kind: "Nil" },
      },
    }
  );
  assert.deepEqual(
    list_range(
      {
        kind: "S",
        f0: {
          kind: "S",
          f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
        },
      },
      { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } }
    ),
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
          f1: { kind: "Nil" },
        },
      },
    }
  );
}
validations();
