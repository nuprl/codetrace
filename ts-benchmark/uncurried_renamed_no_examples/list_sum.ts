declare var require: any;
const assert = require("node:assert");
type _uniq_0 = { kind: "Ctor2" } | { kind: "Ctor3"; f0: _uniq_0 };
type _uniq_1 = { kind: "Ctor4" } | { kind: "Ctor5"; f0: _uniq_0; f1: _uniq_1 };

function _uniq_6(
  _uniq_9: _uniq_1,
  _uniq_10: (__x7: _uniq_0, __x8: _uniq_0) => _uniq_0,
  _uniq_11: _uniq_0
): _uniq_0 {
  switch (_uniq_9.kind) {
    case "Ctor4": {
      return _uniq_11;
    }
    case "Ctor5": {
      let _uniq_13 = _uniq_9.f1;
      let _uniq_12 = _uniq_9.f0;
      return _uniq_6(_uniq_13, _uniq_10, _uniq_10(_uniq_11, _uniq_12));
    }
  }
}
function _uniq_7(_uniq_14: _uniq_0, _uniq_15: _uniq_0): _uniq_0 {
  switch (_uniq_14.kind) {
    case "Ctor2": {
      return _uniq_15;
    }
    case "Ctor3": {
      let _uniq_16 = _uniq_14.f0;
      return { kind: "Ctor3", f0: _uniq_7(_uniq_16, _uniq_15) };
    }
  }
}

function _uniq_8(_uniq_17: _uniq_1): _uniq_0 {
  return _uniq_6(_uniq_17, _uniq_7, { kind: "Ctor2" });
}

function assertions() {
  assert.deepEqual(_uniq_8({ kind: "Ctor4" }), { kind: "Ctor2" });
  assert.deepEqual(
    _uniq_8({
      kind: "Ctor5",
      f0: { kind: "Ctor3", f0: { kind: "Ctor3", f0: { kind: "Ctor2" } } },
      f1: {
        kind: "Ctor5",
        f0: { kind: "Ctor3", f0: { kind: "Ctor2" } },
        f1: { kind: "Ctor4" },
      },
    }),
    {
      kind: "Ctor3",
      f0: { kind: "Ctor3", f0: { kind: "Ctor3", f0: { kind: "Ctor2" } } },
    }
  );
}
assertions();

function validations() {
  assert.deepEqual(
    _uniq_8({ kind: "Ctor5", f0: { kind: "Ctor2" }, f1: { kind: "Ctor4" } }),
    { kind: "Ctor2" }
  );
  assert.deepEqual(
    _uniq_8({
      kind: "Ctor5",
      f0: { kind: "Ctor3", f0: { kind: "Ctor2" } },
      f1: {
        kind: "Ctor5",
        f0: { kind: "Ctor3", f0: { kind: "Ctor3", f0: { kind: "Ctor2" } } },
        f1: { kind: "Ctor4" },
      },
    }),
    {
      kind: "Ctor3",
      f0: { kind: "Ctor3", f0: { kind: "Ctor3", f0: { kind: "Ctor2" } } },
    }
  );
  assert.deepEqual(
    _uniq_8({
      kind: "Ctor5",
      f0: { kind: "Ctor3", f0: { kind: "Ctor3", f0: { kind: "Ctor2" } } },
      f1: {
        kind: "Ctor5",
        f0: { kind: "Ctor3", f0: { kind: "Ctor2" } },
        f1: {
          kind: "Ctor5",
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
          f1: { kind: "Ctor4" },
        },
      },
    }),
    {
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
                f0: {
                  kind: "Ctor3",
                  f0: { kind: "Ctor3", f0: { kind: "Ctor2" } },
                },
              },
            },
          },
        },
      },
    }
  );
}
validations();
