declare var require: any;
const assert = require("node:assert");
type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4"; f0: _uniq_0 };
type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_0; f1: _uniq_1 };
type _uniq_2 =
  | { kind: "Ctor7" }
  | { kind: "Ctor8"; f0: _uniq_2; f1: _uniq_0; f2: _uniq_2 };

function _uniq_9(_uniq_12: _uniq_0): _uniq_0 {
  switch (_uniq_12.kind) {
    case "Ctor3": {
      return { kind: "Ctor3" };
    }
    case "Ctor4": {
      let _uniq_13 = _uniq_12.f0;
      switch (_uniq_13.kind) {
        case "Ctor3": {
          return { kind: "Ctor3" };
        }
        case "Ctor4": {
          let _uniq_14 = _uniq_13.f0;
          return { kind: "Ctor4", f0: _uniq_9(_uniq_14) };
        }
      }
    }
  }
}
function _uniq_10(_uniq_15: _uniq_0): _uniq_0 {
  return { kind: "Ctor4", f0: _uniq_15 };
}

function _uniq_11(
  _uniq_16: (__x14: _uniq_0) => _uniq_0
): (_uniq_17: _uniq_2) => _uniq_2 {
  return function (_uniq_17: _uniq_2) {
    switch (_uniq_17.kind) {
      case "Ctor7": {
        return { kind: "Ctor7" };
      }
      case "Ctor8": {
        let _uniq_20 = _uniq_17.f2;
        let _uniq_19 = _uniq_17.f1;
        let _uniq_18 = _uniq_17.f0;
        return {
          kind: "Ctor8",
          f0: _uniq_11(_uniq_16)(_uniq_18),
          f1: _uniq_16(_uniq_19),
          f2: _uniq_11(_uniq_16)(_uniq_20),
        };
      }
    }
  };
}

function assertions() {
  assert.deepEqual(_uniq_11(_uniq_9)({ kind: "Ctor7" }), { kind: "Ctor7" });
  assert.deepEqual(
    _uniq_11(_uniq_9)({
      kind: "Ctor8",
      f0: {
        kind: "Ctor8",
        f0: { kind: "Ctor7" },
        f1: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
        f2: { kind: "Ctor7" },
      },
      f1: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
      f2: { kind: "Ctor7" },
    }),
    {
      kind: "Ctor8",
      f0: {
        kind: "Ctor8",
        f0: { kind: "Ctor7" },
        f1: { kind: "Ctor4", f0: { kind: "Ctor3" } },
        f2: { kind: "Ctor7" },
      },
      f1: { kind: "Ctor4", f0: { kind: "Ctor3" } },
      f2: { kind: "Ctor7" },
    }
  );
  assert.deepEqual(
    _uniq_11(_uniq_9)({
      kind: "Ctor8",
      f0: { kind: "Ctor7" },
      f1: { kind: "Ctor4", f0: { kind: "Ctor3" } },
      f2: {
        kind: "Ctor8",
        f0: { kind: "Ctor7" },
        f1: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
        f2: { kind: "Ctor7" },
      },
    }),
    {
      kind: "Ctor8",
      f0: { kind: "Ctor7" },
      f1: { kind: "Ctor3" },
      f2: {
        kind: "Ctor8",
        f0: { kind: "Ctor7" },
        f1: { kind: "Ctor4", f0: { kind: "Ctor3" } },
        f2: { kind: "Ctor7" },
      },
    }
  );
  assert.deepEqual(
    _uniq_11(_uniq_10)({
      kind: "Ctor8",
      f0: { kind: "Ctor7" },
      f1: { kind: "Ctor3" },
      f2: { kind: "Ctor7" },
    }),
    {
      kind: "Ctor8",
      f0: { kind: "Ctor7" },
      f1: { kind: "Ctor4", f0: { kind: "Ctor3" } },
      f2: { kind: "Ctor7" },
    }
  );
}
assertions();

function validations() {
  assert.deepEqual(
    _uniq_11(_uniq_10)({
      kind: "Ctor8",
      f0: {
        kind: "Ctor8",
        f0: { kind: "Ctor7" },
        f1: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
        f2: { kind: "Ctor7" },
      },
      f1: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
      f2: { kind: "Ctor7" },
    }),
    {
      kind: "Ctor8",
      f0: {
        kind: "Ctor8",
        f0: { kind: "Ctor7" },
        f1: {
          kind: "Ctor4",
          f0: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
        },
        f2: { kind: "Ctor7" },
      },
      f1: {
        kind: "Ctor4",
        f0: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
      },
      f2: { kind: "Ctor7" },
    }
  );
  assert.deepEqual(
    _uniq_11(function (_uniq_21: _uniq_0) {
      switch (_uniq_21.kind) {
        case "Ctor3": {
          return { kind: "Ctor3" };
        }
        case "Ctor4": {
          let _uniq_22 = _uniq_21.f0;
          return _uniq_22;
        }
      }
    })({
      kind: "Ctor8",
      f0: { kind: "Ctor7" },
      f1: { kind: "Ctor4", f0: { kind: "Ctor3" } },
      f2: {
        kind: "Ctor8",
        f0: {
          kind: "Ctor8",
          f0: { kind: "Ctor7" },
          f1: { kind: "Ctor3" },
          f2: { kind: "Ctor7" },
        },
        f1: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
        f2: {
          kind: "Ctor8",
          f0: { kind: "Ctor7" },
          f1: { kind: "Ctor4", f0: { kind: "Ctor3" } },
          f2: { kind: "Ctor7" },
        },
      },
    }),
    {
      kind: "Ctor8",
      f0: { kind: "Ctor7" },
      f1: { kind: "Ctor3" },
      f2: {
        kind: "Ctor8",
        f0: {
          kind: "Ctor8",
          f0: { kind: "Ctor7" },
          f1: { kind: "Ctor3" },
          f2: { kind: "Ctor7" },
        },
        f1: { kind: "Ctor4", f0: { kind: "Ctor3" } },
        f2: {
          kind: "Ctor8",
          f0: { kind: "Ctor7" },
          f1: { kind: "Ctor3" },
          f2: { kind: "Ctor7" },
        },
      },
    }
  );
  assert.deepEqual(
    _uniq_11(function (_uniq_23: _uniq_0) {
      return _uniq_23;
    })({
      kind: "Ctor8",
      f0: { kind: "Ctor7" },
      f1: {
        kind: "Ctor4",
        f0: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
      },
      f2: { kind: "Ctor7" },
    }),
    {
      kind: "Ctor8",
      f0: { kind: "Ctor7" },
      f1: {
        kind: "Ctor4",
        f0: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
      },
      f2: { kind: "Ctor7" },
    }
  );
}
validations();
