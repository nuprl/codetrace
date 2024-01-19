declare var require: any;
const assert = require("node:assert");
type nat = { kind: "Z" } | { kind: "S"; f0: nat };

function sub(n1: nat, n2: nat): nat {
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
          return sub(n3, n4);
        }
      }
    }
  }
}

function assertions() {
  assert.deepEqual(
    sub(
      { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
      { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } }
    ),
    { kind: "Z" }
  );
  assert.deepEqual(
    sub(
      { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
      { kind: "S", f0: { kind: "Z" } }
    ),
    { kind: "S", f0: { kind: "Z" } }
  );
  assert.deepEqual(
    sub({ kind: "Z" }, { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } }),
    { kind: "Z" }
  );
  assert.deepEqual(
    sub({ kind: "S", f0: { kind: "Z" } }, { kind: "S", f0: { kind: "Z" } }),
    { kind: "Z" }
  );
  assert.deepEqual(sub({ kind: "S", f0: { kind: "Z" } }, { kind: "Z" }), {
    kind: "S",
    f0: { kind: "Z" },
  });
  assert.deepEqual(sub({ kind: "Z" }, { kind: "S", f0: { kind: "Z" } }), {
    kind: "Z",
  });
  assert.deepEqual(
    sub(
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
    { kind: "S", f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } } }
  );
  assert.deepEqual(
    sub(
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
      },
      { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } }
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
assertions();

function validations() {
  assert.deepEqual(
    sub(
      {
        kind: "S",
        f0: {
          kind: "S",
          f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
        },
      },
      { kind: "S", f0: { kind: "Z" } }
    ),
    { kind: "S", f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } } }
  );
  assert.deepEqual(
    sub(
      { kind: "S", f0: { kind: "Z" } },
      {
        kind: "S",
        f0: {
          kind: "S",
          f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
        },
      }
    ),
    { kind: "Z" }
  );
  assert.deepEqual(
    sub({ kind: "S", f0: { kind: "S", f0: { kind: "Z" } } }, { kind: "Z" }),
    { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } }
  );
  assert.deepEqual(
    sub(
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
    ),
    { kind: "Z" }
  );
}
validations();
