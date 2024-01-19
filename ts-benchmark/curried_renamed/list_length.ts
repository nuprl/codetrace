declare var require: any;
const assert = require("node:assert");
type _uniq_0 = { kind: "Ctor2" } | { kind: "Ctor3"; f0: _uniq_0 };
type _uniq_1 = { kind: "Ctor4" } | { kind: "Ctor5"; f0: _uniq_0; f1: _uniq_1 };

function _uniq_6(_uniq_7: _uniq_1): _uniq_0 {
  switch (_uniq_7.kind) {
    case "Ctor4": {
      return { kind: "Ctor2" };
    }
    case "Ctor5": {
      let _uniq_9 = _uniq_7.f1;
      let _uniq_8 = _uniq_7.f0;
      return { kind: "Ctor3", f0: _uniq_6(_uniq_9) };
    }
  }
}

function assertions() {
  assert.deepEqual(_uniq_6({ kind: "Ctor4" }), { kind: "Ctor2" });
  assert.deepEqual(
    _uniq_6({ kind: "Ctor5", f0: { kind: "Ctor2" }, f1: { kind: "Ctor4" } }),
    { kind: "Ctor3", f0: { kind: "Ctor2" } }
  );
  assert.deepEqual(
    _uniq_6({
      kind: "Ctor5",
      f0: { kind: "Ctor2" },
      f1: { kind: "Ctor5", f0: { kind: "Ctor2" }, f1: { kind: "Ctor4" } },
    }),
    { kind: "Ctor3", f0: { kind: "Ctor3", f0: { kind: "Ctor2" } } }
  );
}
assertions();

function validations() {
  assert.deepEqual(
    _uniq_6({
      kind: "Ctor5",
      f0: {
        kind: "Ctor3",
        f0: {
          kind: "Ctor3",
          f0: { kind: "Ctor3", f0: { kind: "Ctor3", f0: { kind: "Ctor2" } } },
        },
      },
      f1: {
        kind: "Ctor5",
        f0: { kind: "Ctor3", f0: { kind: "Ctor2" } },
        f1: {
          kind: "Ctor5",
          f0: {
            kind: "Ctor3",
            f0: { kind: "Ctor3", f0: { kind: "Ctor3", f0: { kind: "Ctor2" } } },
          },
          f1: { kind: "Ctor4" },
        },
      },
    }),
    {
      kind: "Ctor3",
      f0: { kind: "Ctor3", f0: { kind: "Ctor3", f0: { kind: "Ctor2" } } },
    }
  );
  assert.deepEqual(
    _uniq_6({
      kind: "Ctor5",
      f0: { kind: "Ctor3", f0: { kind: "Ctor2" } },
      f1: {
        kind: "Ctor5",
        f0: { kind: "Ctor3", f0: { kind: "Ctor2" } },
        f1: {
          kind: "Ctor5",
          f0: { kind: "Ctor3", f0: { kind: "Ctor2" } },
          f1: {
            kind: "Ctor5",
            f0: { kind: "Ctor3", f0: { kind: "Ctor2" } },
            f1: { kind: "Ctor4" },
          },
        },
      },
    }),
    {
      kind: "Ctor3",
      f0: {
        kind: "Ctor3",
        f0: { kind: "Ctor3", f0: { kind: "Ctor3", f0: { kind: "Ctor2" } } },
      },
    }
  );
}
validations();
