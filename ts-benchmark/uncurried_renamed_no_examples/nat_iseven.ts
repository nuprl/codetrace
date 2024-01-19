declare var require: any;
const assert = require("node:assert");
type _uniq_0 = { kind: "Ctor2" } | { kind: "Ctor3"; f0: _uniq_0 };
type _uniq_1 = { kind: "Ctor4" } | { kind: "Ctor5" };

function _uniq_6(_uniq_7: _uniq_0): _uniq_1 {
  switch (_uniq_7.kind) {
    case "Ctor2": {
      return { kind: "Ctor4" };
    }
    case "Ctor3": {
      let _uniq_8 = _uniq_7.f0;
      switch (_uniq_8.kind) {
        case "Ctor2": {
          return { kind: "Ctor5" };
        }
        case "Ctor3": {
          let _uniq_9 = _uniq_8.f0;
          return _uniq_6(_uniq_9);
        }
      }
    }
  }
}

function assertions() {
  assert.deepEqual(_uniq_6({ kind: "Ctor2" }), { kind: "Ctor4" });
  assert.deepEqual(_uniq_6({ kind: "Ctor3", f0: { kind: "Ctor2" } }), {
    kind: "Ctor5",
  });
  assert.deepEqual(
    _uniq_6({ kind: "Ctor3", f0: { kind: "Ctor3", f0: { kind: "Ctor2" } } }),
    { kind: "Ctor4" }
  );
  assert.deepEqual(
    _uniq_6({
      kind: "Ctor3",
      f0: { kind: "Ctor3", f0: { kind: "Ctor3", f0: { kind: "Ctor2" } } },
    }),
    { kind: "Ctor5" }
  );
}
assertions();

function validations() {
  assert.deepEqual(
    _uniq_6({
      kind: "Ctor3",
      f0: {
        kind: "Ctor3",
        f0: { kind: "Ctor3", f0: { kind: "Ctor3", f0: { kind: "Ctor2" } } },
      },
    }),
    { kind: "Ctor4" }
  );
  assert.deepEqual(
    _uniq_6({
      kind: "Ctor3",
      f0: {
        kind: "Ctor3",
        f0: {
          kind: "Ctor3",
          f0: { kind: "Ctor3", f0: { kind: "Ctor3", f0: { kind: "Ctor2" } } },
        },
      },
    }),
    { kind: "Ctor5" }
  );
  assert.deepEqual(
    _uniq_6({
      kind: "Ctor3",
      f0: {
        kind: "Ctor3",
        f0: {
          kind: "Ctor3",
          f0: {
            kind: "Ctor3",
            f0: { kind: "Ctor3", f0: { kind: "Ctor3", f0: { kind: "Ctor2" } } },
          },
        },
      },
    }),
    { kind: "Ctor4" }
  );
  assert.deepEqual(
    _uniq_6({
      kind: "Ctor3",
      f0: {
        kind: "Ctor3",
        f0: {
          kind: "Ctor3",
          f0: {
            kind: "Ctor3",
            f0: {
              kind: "Ctor3",
              f0: {
                kind: "Ctor3",
                f0: { kind: "Ctor3", f0: { kind: "Ctor2" } },
              },
            },
          },
        },
      },
    }),
    { kind: "Ctor5" }
  );
}
validations();
