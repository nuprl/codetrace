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
function insert(l: nlist, n: nat): nlist {
  switch (l.kind) {
    case "Nil": {
      return { kind: "Cons", f0: n, f1: { kind: "Nil" } };
    }
    case "Cons": {
      let tl = l.f1;
      let m = l.f0;
      switch (compare(n, m).kind) {
        case "LT": {
          return { kind: "Cons", f0: n, f1: { kind: "Cons", f0: m, f1: tl } };
        }
        case "EQ": {
          return l;
        }
        case "GT": {
          return { kind: "Cons", f0: m, f1: insert(tl, n) };
        }
      }
    }
  }
}

function list_sort_sorted_insert(l: nlist): nlist {
  switch (l.kind) {
    case "Nil": {
      return { kind: "Nil" };
    }
    case "Cons": {
      let xs = l.f1;
      let x = l.f0;
      return insert(list_sort_sorted_insert(xs), x);
    }
  }
}

function assertions() {
  assert.deepEqual(list_sort_sorted_insert({ kind: "Nil" }), { kind: "Nil" });
  assert.deepEqual(
    list_sort_sorted_insert({
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
    list_sort_sorted_insert({
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
}
assertions();

function validations() {
  assert.deepEqual(
    list_sort_sorted_insert({
      kind: "Cons",
      f0: { kind: "Z" },
      f1: { kind: "Nil" },
    }),
    { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } }
  );
  assert.deepEqual(
    list_sort_sorted_insert({
      kind: "Cons",
      f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
      f1: { kind: "Cons", f0: { kind: "Z" }, f1: { kind: "Nil" } },
    }),
    {
      kind: "Cons",
      f0: { kind: "Z" },
      f1: {
        kind: "Cons",
        f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
        f1: { kind: "Nil" },
      },
    }
  );
  assert.deepEqual(
    list_sort_sorted_insert({
      kind: "Cons",
      f0: {
        kind: "S",
        f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
      },
      f1: {
        kind: "Cons",
        f0: { kind: "Z" },
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
  );
  assert.deepEqual(
    list_sort_sorted_insert({
      kind: "Cons",
      f0: {
        kind: "S",
        f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
      },
      f1: {
        kind: "Cons",
        f0: { kind: "Z" },
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
      kind: "Cons",
      f0: { kind: "Z" },
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
}
validations();
