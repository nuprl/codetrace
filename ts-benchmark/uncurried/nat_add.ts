declare var require: any;
const assert = require("node:assert");
type nat = { kind: "Z" } | { kind: "S"; f0: nat };

function add(x: nat, y: nat): nat {
  switch (x.kind) {
    case "Z": {
      return y;
    }
    case "S": {
      let xp = x.f0;
      return { kind: "S", f0: add(xp, y) };
    }
  }
}

function assertions() {
  assert.deepEqual(add({ kind: "Z" }, { kind: "Z" }), { kind: "Z" });
  assert.deepEqual(add({ kind: "Z" }, { kind: "S", f0: { kind: "Z" } }), {
    kind: "S",
    f0: { kind: "Z" },
  });
  assert.deepEqual(add({ kind: "S", f0: { kind: "Z" } }, { kind: "Z" }), {
    kind: "S",
    f0: { kind: "Z" },
  });
  assert.deepEqual(
    add({ kind: "Z" }, { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } }),
    { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } }
  );
  assert.deepEqual(
    add({ kind: "S", f0: { kind: "Z" } }, { kind: "S", f0: { kind: "Z" } }),
    { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } }
  );
  assert.deepEqual(
    add({ kind: "S", f0: { kind: "S", f0: { kind: "Z" } } }, { kind: "Z" }),
    { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } }
  );
  assert.deepEqual(
    add(
      { kind: "S", f0: { kind: "Z" } },
      { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } }
    ),
    { kind: "S", f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } } }
  );
  assert.deepEqual(
    add(
      { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
      { kind: "S", f0: { kind: "Z" } }
    ),
    { kind: "S", f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } } }
  );
}
assertions();

function validations() {
  assert.deepEqual(
    add(
      { kind: "S", f0: { kind: "Z" } },
      { kind: "S", f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } } }
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
    add(
      { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
      { kind: "S", f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } } }
    ),
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
  assert.deepEqual(
    add(
      { kind: "Z" },
      { kind: "S", f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } } }
    ),
    { kind: "S", f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } } }
  );
  assert.deepEqual(
    add(
      { kind: "S", f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } } },
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
            f0: { kind: "S", f0: { kind: "S", f0: { kind: "Z" } } },
          },
        },
      },
    }
  );
}
validations();
