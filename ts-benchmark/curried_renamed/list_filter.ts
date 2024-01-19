declare var require: any;
const assert = require("node:assert");
type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4"; f0: _uniq_0 };
type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_0; f1: _uniq_1 };
type _uniq_2 = { kind: "Ctor7" } | { kind: "Ctor8" };

function _uniq_9(_uniq_12: _uniq_0): _uniq_2 {
  switch (_uniq_12.kind) {
    case "Ctor3": {
      return { kind: "Ctor7" };
    }
    case "Ctor4": {
      let _uniq_13 = _uniq_12.f0;
      switch (_uniq_13.kind) {
        case "Ctor3": {
          return { kind: "Ctor8" };
        }
        case "Ctor4": {
          let _uniq_14 = _uniq_13.f0;
          return _uniq_9(_uniq_14);
        }
      }
    }
  }
}
function _uniq_10(_uniq_15: _uniq_0): _uniq_2 {
  switch (_uniq_15.kind) {
    case "Ctor3": {
      return { kind: "Ctor8" };
    }
    case "Ctor4": {
      let _uniq_16 = _uniq_15.f0;
      return { kind: "Ctor7" };
    }
  }
}

function _uniq_11(
  _uniq_17: (__x0: _uniq_0) => _uniq_2
): (_uniq_18: _uniq_1) => _uniq_1 {
  return function (_uniq_18: _uniq_1) {
    switch (_uniq_18.kind) {
      case "Ctor5": {
        return { kind: "Ctor5" };
      }
      case "Ctor6": {
        let _uniq_20 = _uniq_18.f1;
        let _uniq_19 = _uniq_18.f0;
        switch (_uniq_17(_uniq_19).kind) {
          case "Ctor7": {
            return {
              kind: "Ctor6",
              f0: _uniq_19,
              f1: _uniq_11(_uniq_17)(_uniq_20),
            };
          }
          case "Ctor8": {
            return _uniq_11(_uniq_17)(_uniq_20);
          }
        }
      }
    }
  };
}

function assertions() {
  assert.deepEqual(_uniq_11(_uniq_9)({ kind: "Ctor5" }), { kind: "Ctor5" });
  assert.deepEqual(
    _uniq_11(_uniq_9)({
      kind: "Ctor6",
      f0: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
      f1: { kind: "Ctor5" },
    }),
    {
      kind: "Ctor6",
      f0: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
      f1: { kind: "Ctor5" },
    }
  );
  assert.deepEqual(
    _uniq_11(_uniq_9)({
      kind: "Ctor6",
      f0: { kind: "Ctor3" },
      f1: { kind: "Ctor6", f0: { kind: "Ctor3" }, f1: { kind: "Ctor5" } },
    }),
    {
      kind: "Ctor6",
      f0: { kind: "Ctor3" },
      f1: { kind: "Ctor6", f0: { kind: "Ctor3" }, f1: { kind: "Ctor5" } },
    }
  );
  assert.deepEqual(
    _uniq_11(_uniq_9)({
      kind: "Ctor6",
      f0: { kind: "Ctor4", f0: { kind: "Ctor3" } },
      f1: {
        kind: "Ctor6",
        f0: { kind: "Ctor3" },
        f1: {
          kind: "Ctor6",
          f0: { kind: "Ctor4", f0: { kind: "Ctor3" } },
          f1: { kind: "Ctor5" },
        },
      },
    }),
    { kind: "Ctor6", f0: { kind: "Ctor3" }, f1: { kind: "Ctor5" } }
  );
  assert.deepEqual(
    _uniq_11(_uniq_10)({
      kind: "Ctor6",
      f0: { kind: "Ctor3" },
      f1: { kind: "Ctor5" },
    }),
    { kind: "Ctor5" }
  );
}
assertions();

function validations() {
  assert.deepEqual(
    _uniq_11(_uniq_9)({
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
      kind: "Ctor6",
      f0: {
        kind: "Ctor4",
        f0: {
          kind: "Ctor4",
          f0: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
        },
      },
      f1: { kind: "Ctor5" },
    }
  );
  assert.deepEqual(
    _uniq_11(_uniq_9)({
      kind: "Ctor6",
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
      f1: {
        kind: "Ctor6",
        f0: { kind: "Ctor4", f0: { kind: "Ctor3" } },
        f1: {
          kind: "Ctor6",
          f0: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
          f1: { kind: "Ctor5" },
        },
      },
    }),
    {
      kind: "Ctor6",
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
      f1: {
        kind: "Ctor6",
        f0: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
        f1: { kind: "Ctor5" },
      },
    }
  );
  assert.deepEqual(
    _uniq_11(_uniq_9)({
      kind: "Ctor6",
      f0: { kind: "Ctor3" },
      f1: {
        kind: "Ctor6",
        f0: { kind: "Ctor4", f0: { kind: "Ctor3" } },
        f1: {
          kind: "Ctor6",
          f0: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
          f1: {
            kind: "Ctor6",
            f0: {
              kind: "Ctor4",
              f0: {
                kind: "Ctor4",
                f0: { kind: "Ctor4", f0: { kind: "Ctor3" } },
              },
            },
            f1: { kind: "Ctor5" },
          },
        },
      },
    }),
    {
      kind: "Ctor6",
      f0: { kind: "Ctor3" },
      f1: {
        kind: "Ctor6",
        f0: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
        f1: { kind: "Ctor5" },
      },
    }
  );
  assert.deepEqual(
    _uniq_11(_uniq_10)({
      kind: "Ctor6",
      f0: { kind: "Ctor3" },
      f1: {
        kind: "Ctor6",
        f0: { kind: "Ctor4", f0: { kind: "Ctor3" } },
        f1: {
          kind: "Ctor6",
          f0: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
          f1: {
            kind: "Ctor6",
            f0: {
              kind: "Ctor4",
              f0: {
                kind: "Ctor4",
                f0: { kind: "Ctor4", f0: { kind: "Ctor3" } },
              },
            },
            f1: { kind: "Ctor5" },
          },
        },
      },
    }),
    {
      kind: "Ctor6",
      f0: { kind: "Ctor4", f0: { kind: "Ctor3" } },
      f1: {
        kind: "Ctor6",
        f0: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
        f1: {
          kind: "Ctor6",
          f0: {
            kind: "Ctor4",
            f0: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
          },
          f1: { kind: "Ctor5" },
        },
      },
    }
  );
  assert.deepEqual(
    _uniq_11(_uniq_10)({
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
    }
  );
}
validations();
