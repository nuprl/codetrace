declare var require: any;
const assert = require("node:assert");
type nat = { kind: "Z" } | { kind: "S"; f0: nat };
type nlist = { kind: "Nil" } | { kind: "Cons"; f0: nat; f1: nlist };
type bbool = { kind: "True" } | { kind: "False" };

function sum(n1: nat, n2: nat): nat {
  switch (n1.kind) {
    case "Z": {
      return n2;
    }
    case "S": {
      let n3 = n1.f0;
      return { kind: "S", f0: sum(n3, n2) };
    }
  }
}
function is_odd(n: nat): bbool {
  switch (n.kind) {
    case "Z": {
      return { kind: "False" };
    }
    case "S": {
      let n1 = n.f0;
      switch (n1.kind) {
        case "Z": {
          return { kind: "True" };
        }
        case "S": {
          let n2 = n1.f0;
          return is_odd(n2);
        }
      }
    }
  }
}
function count_odd(n1: nat, n2: nat): nat {
  switch (is_odd(n2).kind) {
    case "True": {
      return { kind: "S", f0: n1 };
    }
    case "False": {
      return n1;
    }
  }
}

function list_fold(f: (__x1: nat, __x2: nat) => nat, acc: nat, l: nlist): nat {
  switch (l.kind) {
    case "Nil": {
      return acc;
    }
    case "Cons": {
      let xs = l.f1;
      let x = l.f0;
      return list_fold(f, f(acc, x), xs);
    }
  }
}

function assertions() {
  assert.deepEqual(list_fold(sum, { kind: "Z" }, { kind: "Nil" }), {
    kind: "Z",
  });
  assert.deepEqual(
    list_fold(
      sum,
      { kind: "Z" },
      {
        kind: "Cons",
        f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
        f1: {
          kind: "Cons",
          f0: { kind: "S", f0: { kind: "Z" } },
          f1: { kind: "Nil" },
        },
      }
    ),
    { kind: "S", f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } } }
  );
  assert.deepEqual(
    list_fold(
      sum,
      { kind: "Z" },
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
}
assertions();

function validations() {
  assert.deepEqual(
    list_fold(
      sum,
      { kind: "Z" },
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
                f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
              },
            },
          },
        },
      },
    }
  );
  assert.deepEqual(
    list_fold(
      sum,
      { kind: "Z" },
      {
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
      }
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
    list_fold(
      sum,
      { kind: "Z" },
      {
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
      }
    ),
    { kind: "Z" }
  );
}
validations();
