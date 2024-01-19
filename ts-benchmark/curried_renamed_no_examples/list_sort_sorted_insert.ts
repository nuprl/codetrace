declare var require: any;
const assert = require("node:assert");
type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4"; f0: _uniq_0 };
type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_0; f1: _uniq_1 };
type _uniq_2 = { kind: "Ctor7" } | { kind: "Ctor8" } | { kind: "Ctor9" };

function _uniq_10(_uniq_13: _uniq_0): (_uniq_14: _uniq_0) => _uniq_2 {
  return function (_uniq_14: _uniq_0) {
    switch (_uniq_13.kind) {
      case "Ctor3": {
        switch (_uniq_14.kind) {
          case "Ctor3": {
            return { kind: "Ctor8" };
          }
          case "Ctor4": {
            let _uniq_15 = _uniq_14.f0;
            return { kind: "Ctor7" };
          }
        }
      }
      case "Ctor4": {
        let _uniq_16 = _uniq_13.f0;
        switch (_uniq_14.kind) {
          case "Ctor3": {
            return { kind: "Ctor9" };
          }
          case "Ctor4": {
            let _uniq_17 = _uniq_14.f0;
            return _uniq_10(_uniq_16)(_uniq_17);
          }
        }
      }
    }
  };
}
function _uniq_11(_uniq_18: _uniq_1): (_uniq_19: _uniq_0) => _uniq_1 {
  return function (_uniq_19: _uniq_0) {
    switch (_uniq_18.kind) {
      case "Ctor5": {
        return { kind: "Ctor6", f0: _uniq_19, f1: { kind: "Ctor5" } };
      }
      case "Ctor6": {
        let _uniq_21 = _uniq_18.f1;
        let _uniq_20 = _uniq_18.f0;
        switch (_uniq_10(_uniq_19)(_uniq_20).kind) {
          case "Ctor7": {
            return {
              kind: "Ctor6",
              f0: _uniq_19,
              f1: { kind: "Ctor6", f0: _uniq_20, f1: _uniq_21 },
            };
          }
          case "Ctor8": {
            return _uniq_18;
          }
          case "Ctor9": {
            return {
              kind: "Ctor6",
              f0: _uniq_20,
              f1: _uniq_11(_uniq_21)(_uniq_19),
            };
          }
        }
      }
    }
  };
}

function _uniq_12(_uniq_22: _uniq_1): _uniq_1 {
  switch (_uniq_22.kind) {
    case "Ctor5": {
      return { kind: "Ctor5" };
    }
    case "Ctor6": {
      let _uniq_24 = _uniq_22.f1;
      let _uniq_23 = _uniq_22.f0;
      return _uniq_11(_uniq_12(_uniq_24))(_uniq_23);
    }
  }
}

function assertions() {
  assert.deepEqual(_uniq_12({ kind: "Ctor5" }), { kind: "Ctor5" });
  assert.deepEqual(
    _uniq_12({
      kind: "Ctor6",
      f0: { kind: "Ctor4", f0: { kind: "Ctor3" } },
      f1: {
        kind: "Ctor6",
        f0: { kind: "Ctor4", f0: { kind: "Ctor3" } },
        f1: { kind: "Ctor5" },
      },
    }),
    {
      kind: "Ctor6",
      f0: { kind: "Ctor4", f0: { kind: "Ctor3" } },
      f1: { kind: "Ctor5" },
    }
  );
  assert.deepEqual(
    _uniq_12({
      kind: "Ctor6",
      f0: { kind: "Ctor3" },
      f1: {
        kind: "Ctor6",
        f0: { kind: "Ctor4", f0: { kind: "Ctor3" } },
        f1: {
          kind: "Ctor6",
          f0: { kind: "Ctor4", f0: { kind: "Ctor3" } },
          f1: { kind: "Ctor5" },
        },
      },
    }),
    {
      kind: "Ctor6",
      f0: { kind: "Ctor3" },
      f1: {
        kind: "Ctor6",
        f0: { kind: "Ctor4", f0: { kind: "Ctor3" } },
        f1: { kind: "Ctor5" },
      },
    }
  );
}
assertions();

function validations() {
  assert.deepEqual(
    _uniq_12({ kind: "Ctor6", f0: { kind: "Ctor3" }, f1: { kind: "Ctor5" } }),
    { kind: "Ctor6", f0: { kind: "Ctor3" }, f1: { kind: "Ctor5" } }
  );
  assert.deepEqual(
    _uniq_12({
      kind: "Ctor6",
      f0: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
      f1: { kind: "Ctor6", f0: { kind: "Ctor3" }, f1: { kind: "Ctor5" } },
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
    _uniq_12({
      kind: "Ctor6",
      f0: {
        kind: "Ctor4",
        f0: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
      },
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
    {
      kind: "Ctor6",
      f0: { kind: "Ctor3" },
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
    }
  );
  assert.deepEqual(
    _uniq_12({
      kind: "Ctor6",
      f0: {
        kind: "Ctor4",
        f0: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
      },
      f1: {
        kind: "Ctor6",
        f0: { kind: "Ctor3" },
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
      f0: { kind: "Ctor3" },
      f1: {
        kind: "Ctor6",
        f0: {
          kind: "Ctor4",
          f0: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
        },
        f1: { kind: "Ctor5" },
      },
    }
  );
}
validations();
