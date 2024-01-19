declare var require: any;
const assert = require("node:assert");
type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4"; f0: _uniq_0 };
type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_0; f1: _uniq_1 };
type _uniq_2 = { kind: "Ctor7" } | { kind: "Ctor8" };

function _uniq_9(_uniq_13: _uniq_0): (_uniq_14: _uniq_0) => _uniq_0 {
  return function (_uniq_14: _uniq_0) {
    switch (_uniq_13.kind) {
      case "Ctor3": {
        return _uniq_14;
      }
      case "Ctor4": {
        let _uniq_15 = _uniq_13.f0;
        return { kind: "Ctor4", f0: _uniq_9(_uniq_15)(_uniq_14) };
      }
    }
  };
}
function _uniq_10(_uniq_16: _uniq_0): _uniq_2 {
  switch (_uniq_16.kind) {
    case "Ctor3": {
      return { kind: "Ctor8" };
    }
    case "Ctor4": {
      let _uniq_17 = _uniq_16.f0;
      switch (_uniq_17.kind) {
        case "Ctor3": {
          return { kind: "Ctor7" };
        }
        case "Ctor4": {
          let _uniq_18 = _uniq_17.f0;
          return _uniq_10(_uniq_18);
        }
      }
    }
  }
}
function _uniq_11(_uniq_19: _uniq_0): (_uniq_20: _uniq_0) => _uniq_0 {
  return function (_uniq_20: _uniq_0) {
    switch (_uniq_10(_uniq_20).kind) {
      case "Ctor7": {
        return { kind: "Ctor4", f0: _uniq_19 };
      }
      case "Ctor8": {
        return _uniq_19;
      }
    }
  };
}

function _uniq_12(
  _uniq_21: (__x1: _uniq_0) => (__x2: _uniq_0) => _uniq_0
): (_uniq_22: _uniq_0) => (_uniq_23: _uniq_1) => _uniq_0 {
  return function (_uniq_22: _uniq_0) {
    return function (_uniq_23: _uniq_1) {
      switch (_uniq_23.kind) {
        case "Ctor5": {
          return _uniq_22;
        }
        case "Ctor6": {
          let _uniq_25 = _uniq_23.f1;
          let _uniq_24 = _uniq_23.f0;
          return _uniq_12(_uniq_21)(_uniq_21(_uniq_22)(_uniq_24))(_uniq_25);
        }
      }
    };
  };
}

function assertions() {
  assert.deepEqual(_uniq_12(_uniq_9)({ kind: "Ctor3" })({ kind: "Ctor5" }), {
    kind: "Ctor3",
  });
  assert.deepEqual(
    _uniq_12(_uniq_9)({ kind: "Ctor3" })({
      kind: "Ctor6",
      f0: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
      f1: {
        kind: "Ctor6",
        f0: { kind: "Ctor4", f0: { kind: "Ctor3" } },
        f1: { kind: "Ctor5" },
      },
    }),
    {
      kind: "Ctor4",
      f0: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
    }
  );
  assert.deepEqual(
    _uniq_12(_uniq_9)({ kind: "Ctor3" })({
      kind: "Ctor6",
      f0: {
        kind: "Ctor4",
        f0: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
      },
      f1: {
        kind: "Ctor6",
        f0: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
        f1: {
          kind: "Ctor6",
          f0: { kind: "Ctor4", f0: { kind: "Ctor3" } },
          f1: { kind: "Ctor5" },
        },
      },
    }),
    {
      kind: "Ctor4",
      f0: {
        kind: "Ctor4",
        f0: {
          kind: "Ctor4",
          f0: {
            kind: "Ctor4",
            f0: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
          },
        },
      },
    }
  );
}
assertions();

function validations() {
  assert.deepEqual(
    _uniq_12(_uniq_9)({ kind: "Ctor3" })({
      kind: "Ctor6",
      f0: {
        kind: "Ctor4",
        f0: {
          kind: "Ctor4",
          f0: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
        },
      },
      f1: {
        kind: "Ctor6",
        f0: { kind: "Ctor4", f0: { kind: "Ctor3" } },
        f1: {
          kind: "Ctor6",
          f0: {
            kind: "Ctor4",
            f0: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
          },
          f1: { kind: "Ctor5" },
        },
      },
    }),
    {
      kind: "Ctor4",
      f0: {
        kind: "Ctor4",
        f0: {
          kind: "Ctor4",
          f0: {
            kind: "Ctor4",
            f0: {
              kind: "Ctor4",
              f0: {
                kind: "Ctor4",
                f0: {
                  kind: "Ctor4",
                  f0: { kind: "Ctor4", f0: { kind: "Ctor3" } },
                },
              },
            },
          },
        },
      },
    }
  );
  assert.deepEqual(
    _uniq_12(_uniq_9)({ kind: "Ctor3" })({
      kind: "Ctor6",
      f0: { kind: "Ctor4", f0: { kind: "Ctor3" } },
      f1: {
        kind: "Ctor6",
        f0: { kind: "Ctor4", f0: { kind: "Ctor3" } },
        f1: {
          kind: "Ctor6",
          f0: { kind: "Ctor4", f0: { kind: "Ctor3" } },
          f1: {
            kind: "Ctor6",
            f0: { kind: "Ctor4", f0: { kind: "Ctor3" } },
            f1: { kind: "Ctor5" },
          },
        },
      },
    }),
    {
      kind: "Ctor4",
      f0: {
        kind: "Ctor4",
        f0: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
      },
    }
  );
  assert.deepEqual(
    _uniq_12(_uniq_9)({ kind: "Ctor3" })({
      kind: "Ctor6",
      f0: { kind: "Ctor3" },
      f1: {
        kind: "Ctor6",
        f0: { kind: "Ctor3" },
        f1: {
          kind: "Ctor6",
          f0: { kind: "Ctor3" },
          f1: { kind: "Ctor6", f0: { kind: "Ctor3" }, f1: { kind: "Ctor5" } },
        },
      },
    }),
    { kind: "Ctor3" }
  );
}
validations();
