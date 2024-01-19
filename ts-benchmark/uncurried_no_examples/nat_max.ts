declare var require: any;
const assert = require("node:assert");
type nat = { kind: "Z" } | { kind: "S"; f0: nat };
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

function nat_max(n1: nat, n2: nat): nat {
  switch (n1.kind) {
    case "Z": {
      return n2;
    }
    case "S": {
      let n3 = n1.f0;
      switch (n2.kind) {
        case "Z": {
          return n1;
        }
        case "S": {
          let n4 = n2.f0;
          return { kind: "S", f0: nat_max(n3, n4) };
        }
      }
    }
  }
}

function assertions() {
  assert.deepEqual(nat_max({ kind: "Z" }, { kind: "Z" }), { kind: "Z" });
  assert.deepEqual(nat_max({ kind: "Z" }, { kind: "S", f0: { kind: "Z" } }), {
    kind: "S",
    f0: { kind: "Z" },
  });
  assert.deepEqual(
    nat_max({ kind: "Z" }, { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } }),
    { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } }
  );
  assert.deepEqual(nat_max({ kind: "S", f0: { kind: "Z" } }, { kind: "Z" }), {
    kind: "S",
    f0: { kind: "Z" },
  });
  assert.deepEqual(
    nat_max({ kind: "S", f0: { kind: "Z" } }, { kind: "S", f0: { kind: "Z" } }),
    { kind: "S", f0: { kind: "Z" } }
  );
  assert.deepEqual(
    nat_max(
      { kind: "S", f0: { kind: "Z" } },
      { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } }
    ),
    { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } }
  );
  assert.deepEqual(
    nat_max({ kind: "S", f0: { kind: "S", f0: { kind: "Z" } } }, { kind: "Z" }),
    { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } }
  );
  assert.deepEqual(
    nat_max(
      { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
      { kind: "S", f0: { kind: "Z" } }
    ),
    { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } }
  );
  assert.deepEqual(
    nat_max(
      { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
      { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } }
    ),
    { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } }
  );
}
assertions();

function validations() {
  assert.deepEqual(
    nat_max(
      { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
      { kind: "S", f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } } }
    ),
    { kind: "S", f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } } }
  );
  assert.deepEqual(
    nat_max(
      { kind: "S", f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } } },
      { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } }
    ),
    { kind: "S", f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } } }
  );
  assert.deepEqual(
    nat_max(
      { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
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
        f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
      },
    }
  );
  assert.deepEqual(
    nat_max(
      {
        kind: "S",
        f0: {
          kind: "S",
          f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
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
}
validations();
