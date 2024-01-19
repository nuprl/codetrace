declare var require: any;
const assert = require("node:assert");
type nat = { kind: "Z" } | { kind: "S"; f0: nat };
type bbool = { kind: "True" } | { kind: "False" };

function nat_iseven(n: nat): bbool {
  switch (n.kind) {
    case "Z": {
      return { kind: "True" };
    }
    case "S": {
      let n1 = n.f0;
      switch (n1.kind) {
        case "Z": {
          return { kind: "False" };
        }
        case "S": {
          let n2p = n1.f0;
          return nat_iseven(n2p);
        }
      }
    }
  }
}

function assertions() {
  assert.deepEqual(nat_iseven({ kind: "Z" }), { kind: "True" });
  assert.deepEqual(nat_iseven({ kind: "S", f0: { kind: "Z" } }), {
    kind: "False",
  });
  assert.deepEqual(
    nat_iseven({ kind: "S", f0: { kind: "S", f0: { kind: "Z" } } }),
    { kind: "True" }
  );
  assert.deepEqual(
    nat_iseven({
      kind: "S",
      f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
    }),
    { kind: "False" }
  );
}
assertions();

function validations() {
  assert.deepEqual(
    nat_iseven({
      kind: "S",
      f0: {
        kind: "S",
        f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
      },
    }),
    { kind: "True" }
  );
  assert.deepEqual(
    nat_iseven({
      kind: "S",
      f0: {
        kind: "S",
        f0: {
          kind: "S",
          f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
        },
      },
    }),
    { kind: "False" }
  );
  assert.deepEqual(
    nat_iseven({
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
    }),
    { kind: "True" }
  );
  assert.deepEqual(
    nat_iseven({
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
    }),
    { kind: "False" }
  );
}
validations();
