declare var require: any;
const assert = require("node:assert");
type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4" };
type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_1 };
type _uniq_2 =
  | { kind: "Ctor7" }
  | { kind: "Ctor8"; f0: _uniq_2; f1: _uniq_0; f2: _uniq_2 };

function _uniq_9(_uniq_11: _uniq_1): (_uniq_12: _uniq_1) => _uniq_1 {
  return function (_uniq_12: _uniq_1) {
    switch (_uniq_11.kind) {
      case "Ctor5": {
        return _uniq_12;
      }
      case "Ctor6": {
        let _uniq_13 = _uniq_11.f0;
        return { kind: "Ctor6", f0: _uniq_9(_uniq_13)(_uniq_12) };
      }
    }
  };
}

function _uniq_10(_uniq_14: _uniq_2): (_uniq_15: _uniq_1) => _uniq_1 {
  return function (_uniq_15: _uniq_1) {
    switch (_uniq_14.kind) {
      case "Ctor7": {
        return { kind: "Ctor5" };
      }
      case "Ctor8": {
        let _uniq_18 = _uniq_14.f2;
        let _uniq_17 = _uniq_14.f1;
        let _uniq_16 = _uniq_14.f0;
        switch (_uniq_15.kind) {
          case "Ctor5": {
            return { kind: "Ctor6", f0: { kind: "Ctor5" } };
          }
          case "Ctor6": {
            let _uniq_19 = _uniq_15.f0;
            return _uniq_9(_uniq_10(_uniq_16)(_uniq_19))(
              _uniq_10(_uniq_18)(_uniq_19)
            );
          }
        }
      }
    }
  };
}

function assertions() {
  assert.deepEqual(_uniq_10({ kind: "Ctor7" })({ kind: "Ctor5" }), {
    kind: "Ctor5",
  });
  assert.deepEqual(
    _uniq_10({ kind: "Ctor7" })({ kind: "Ctor6", f0: { kind: "Ctor5" } }),
    { kind: "Ctor5" }
  );
  assert.deepEqual(
    _uniq_10({ kind: "Ctor7" })({
      kind: "Ctor6",
      f0: { kind: "Ctor6", f0: { kind: "Ctor5" } },
    }),
    { kind: "Ctor5" }
  );
  assert.deepEqual(
    _uniq_10({ kind: "Ctor7" })({
      kind: "Ctor6",
      f0: { kind: "Ctor6", f0: { kind: "Ctor6", f0: { kind: "Ctor5" } } },
    }),
    { kind: "Ctor5" }
  );
  assert.deepEqual(
    _uniq_10({
      kind: "Ctor8",
      f0: { kind: "Ctor7" },
      f1: { kind: "Ctor3" },
      f2: { kind: "Ctor7" },
    })({ kind: "Ctor5" }),
    { kind: "Ctor6", f0: { kind: "Ctor5" } }
  );
  assert.deepEqual(
    _uniq_10({
      kind: "Ctor8",
      f0: { kind: "Ctor7" },
      f1: { kind: "Ctor3" },
      f2: { kind: "Ctor7" },
    })({ kind: "Ctor6", f0: { kind: "Ctor5" } }),
    { kind: "Ctor5" }
  );
  assert.deepEqual(
    _uniq_10({
      kind: "Ctor8",
      f0: { kind: "Ctor7" },
      f1: { kind: "Ctor3" },
      f2: { kind: "Ctor7" },
    })({ kind: "Ctor6", f0: { kind: "Ctor6", f0: { kind: "Ctor5" } } }),
    { kind: "Ctor5" }
  );
  assert.deepEqual(
    _uniq_10({
      kind: "Ctor8",
      f0: { kind: "Ctor7" },
      f1: { kind: "Ctor3" },
      f2: { kind: "Ctor7" },
    })({
      kind: "Ctor6",
      f0: { kind: "Ctor6", f0: { kind: "Ctor6", f0: { kind: "Ctor5" } } },
    }),
    { kind: "Ctor5" }
  );
  assert.deepEqual(
    _uniq_10({
      kind: "Ctor8",
      f0: {
        kind: "Ctor8",
        f0: { kind: "Ctor7" },
        f1: { kind: "Ctor3" },
        f2: { kind: "Ctor7" },
      },
      f1: { kind: "Ctor3" },
      f2: { kind: "Ctor7" },
    })({ kind: "Ctor5" }),
    { kind: "Ctor6", f0: { kind: "Ctor5" } }
  );
  assert.deepEqual(
    _uniq_10({
      kind: "Ctor8",
      f0: {
        kind: "Ctor8",
        f0: { kind: "Ctor7" },
        f1: { kind: "Ctor3" },
        f2: { kind: "Ctor7" },
      },
      f1: { kind: "Ctor3" },
      f2: { kind: "Ctor7" },
    })({ kind: "Ctor6", f0: { kind: "Ctor5" } }),
    { kind: "Ctor6", f0: { kind: "Ctor5" } }
  );
  assert.deepEqual(
    _uniq_10({
      kind: "Ctor8",
      f0: {
        kind: "Ctor8",
        f0: { kind: "Ctor7" },
        f1: { kind: "Ctor3" },
        f2: { kind: "Ctor7" },
      },
      f1: { kind: "Ctor3" },
      f2: { kind: "Ctor7" },
    })({ kind: "Ctor6", f0: { kind: "Ctor6", f0: { kind: "Ctor5" } } }),
    { kind: "Ctor5" }
  );
  assert.deepEqual(
    _uniq_10({
      kind: "Ctor8",
      f0: {
        kind: "Ctor8",
        f0: { kind: "Ctor7" },
        f1: { kind: "Ctor3" },
        f2: { kind: "Ctor7" },
      },
      f1: { kind: "Ctor3" },
      f2: { kind: "Ctor7" },
    })({
      kind: "Ctor6",
      f0: { kind: "Ctor6", f0: { kind: "Ctor6", f0: { kind: "Ctor5" } } },
    }),
    { kind: "Ctor5" }
  );
  assert.deepEqual(
    _uniq_10({
      kind: "Ctor8",
      f0: { kind: "Ctor7" },
      f1: { kind: "Ctor3" },
      f2: {
        kind: "Ctor8",
        f0: { kind: "Ctor7" },
        f1: { kind: "Ctor3" },
        f2: { kind: "Ctor7" },
      },
    })({ kind: "Ctor5" }),
    { kind: "Ctor6", f0: { kind: "Ctor5" } }
  );
  assert.deepEqual(
    _uniq_10({
      kind: "Ctor8",
      f0: { kind: "Ctor7" },
      f1: { kind: "Ctor3" },
      f2: {
        kind: "Ctor8",
        f0: { kind: "Ctor7" },
        f1: { kind: "Ctor3" },
        f2: { kind: "Ctor7" },
      },
    })({ kind: "Ctor6", f0: { kind: "Ctor5" } }),
    { kind: "Ctor6", f0: { kind: "Ctor5" } }
  );
  assert.deepEqual(
    _uniq_10({
      kind: "Ctor8",
      f0: { kind: "Ctor7" },
      f1: { kind: "Ctor3" },
      f2: {
        kind: "Ctor8",
        f0: { kind: "Ctor7" },
        f1: { kind: "Ctor3" },
        f2: { kind: "Ctor7" },
      },
    })({ kind: "Ctor6", f0: { kind: "Ctor6", f0: { kind: "Ctor5" } } }),
    { kind: "Ctor5" }
  );
  assert.deepEqual(
    _uniq_10({
      kind: "Ctor8",
      f0: { kind: "Ctor7" },
      f1: { kind: "Ctor3" },
      f2: {
        kind: "Ctor8",
        f0: { kind: "Ctor7" },
        f1: { kind: "Ctor3" },
        f2: { kind: "Ctor7" },
      },
    })({
      kind: "Ctor6",
      f0: { kind: "Ctor6", f0: { kind: "Ctor6", f0: { kind: "Ctor5" } } },
    }),
    { kind: "Ctor5" }
  );
  assert.deepEqual(
    _uniq_10({
      kind: "Ctor8",
      f0: {
        kind: "Ctor8",
        f0: { kind: "Ctor7" },
        f1: { kind: "Ctor3" },
        f2: { kind: "Ctor7" },
      },
      f1: { kind: "Ctor3" },
      f2: {
        kind: "Ctor8",
        f0: { kind: "Ctor7" },
        f1: { kind: "Ctor3" },
        f2: { kind: "Ctor7" },
      },
    })({ kind: "Ctor5" }),
    { kind: "Ctor6", f0: { kind: "Ctor5" } }
  );
  assert.deepEqual(
    _uniq_10({
      kind: "Ctor8",
      f0: {
        kind: "Ctor8",
        f0: { kind: "Ctor7" },
        f1: { kind: "Ctor3" },
        f2: { kind: "Ctor7" },
      },
      f1: { kind: "Ctor3" },
      f2: {
        kind: "Ctor8",
        f0: { kind: "Ctor7" },
        f1: { kind: "Ctor3" },
        f2: { kind: "Ctor7" },
      },
    })({ kind: "Ctor6", f0: { kind: "Ctor5" } }),
    { kind: "Ctor6", f0: { kind: "Ctor6", f0: { kind: "Ctor5" } } }
  );
  assert.deepEqual(
    _uniq_10({
      kind: "Ctor8",
      f0: {
        kind: "Ctor8",
        f0: { kind: "Ctor7" },
        f1: { kind: "Ctor3" },
        f2: { kind: "Ctor7" },
      },
      f1: { kind: "Ctor3" },
      f2: {
        kind: "Ctor8",
        f0: { kind: "Ctor7" },
        f1: { kind: "Ctor3" },
        f2: { kind: "Ctor7" },
      },
    })({ kind: "Ctor6", f0: { kind: "Ctor6", f0: { kind: "Ctor5" } } }),
    { kind: "Ctor5" }
  );
  assert.deepEqual(
    _uniq_10({
      kind: "Ctor8",
      f0: {
        kind: "Ctor8",
        f0: { kind: "Ctor7" },
        f1: { kind: "Ctor3" },
        f2: { kind: "Ctor7" },
      },
      f1: { kind: "Ctor3" },
      f2: {
        kind: "Ctor8",
        f0: { kind: "Ctor7" },
        f1: { kind: "Ctor3" },
        f2: { kind: "Ctor7" },
      },
    })({
      kind: "Ctor6",
      f0: { kind: "Ctor6", f0: { kind: "Ctor6", f0: { kind: "Ctor5" } } },
    }),
    { kind: "Ctor5" }
  );
  assert.deepEqual(
    _uniq_10({
      kind: "Ctor8",
      f0: {
        kind: "Ctor8",
        f0: {
          kind: "Ctor8",
          f0: { kind: "Ctor7" },
          f1: { kind: "Ctor3" },
          f2: { kind: "Ctor7" },
        },
        f1: { kind: "Ctor3" },
        f2: {
          kind: "Ctor8",
          f0: { kind: "Ctor7" },
          f1: { kind: "Ctor3" },
          f2: { kind: "Ctor7" },
        },
      },
      f1: { kind: "Ctor3" },
      f2: { kind: "Ctor7" },
    })({ kind: "Ctor5" }),
    { kind: "Ctor6", f0: { kind: "Ctor5" } }
  );
  assert.deepEqual(
    _uniq_10({
      kind: "Ctor8",
      f0: {
        kind: "Ctor8",
        f0: {
          kind: "Ctor8",
          f0: { kind: "Ctor7" },
          f1: { kind: "Ctor3" },
          f2: { kind: "Ctor7" },
        },
        f1: { kind: "Ctor3" },
        f2: {
          kind: "Ctor8",
          f0: { kind: "Ctor7" },
          f1: { kind: "Ctor3" },
          f2: { kind: "Ctor7" },
        },
      },
      f1: { kind: "Ctor3" },
      f2: { kind: "Ctor7" },
    })({ kind: "Ctor6", f0: { kind: "Ctor5" } }),
    { kind: "Ctor6", f0: { kind: "Ctor5" } }
  );
  assert.deepEqual(
    _uniq_10({
      kind: "Ctor8",
      f0: {
        kind: "Ctor8",
        f0: {
          kind: "Ctor8",
          f0: { kind: "Ctor7" },
          f1: { kind: "Ctor3" },
          f2: { kind: "Ctor7" },
        },
        f1: { kind: "Ctor3" },
        f2: {
          kind: "Ctor8",
          f0: { kind: "Ctor7" },
          f1: { kind: "Ctor3" },
          f2: { kind: "Ctor7" },
        },
      },
      f1: { kind: "Ctor3" },
      f2: { kind: "Ctor7" },
    })({ kind: "Ctor6", f0: { kind: "Ctor6", f0: { kind: "Ctor5" } } }),
    { kind: "Ctor6", f0: { kind: "Ctor6", f0: { kind: "Ctor5" } } }
  );
  assert.deepEqual(
    _uniq_10({
      kind: "Ctor8",
      f0: {
        kind: "Ctor8",
        f0: {
          kind: "Ctor8",
          f0: { kind: "Ctor7" },
          f1: { kind: "Ctor3" },
          f2: { kind: "Ctor7" },
        },
        f1: { kind: "Ctor3" },
        f2: {
          kind: "Ctor8",
          f0: { kind: "Ctor7" },
          f1: { kind: "Ctor3" },
          f2: { kind: "Ctor7" },
        },
      },
      f1: { kind: "Ctor3" },
      f2: { kind: "Ctor7" },
    })({
      kind: "Ctor6",
      f0: { kind: "Ctor6", f0: { kind: "Ctor6", f0: { kind: "Ctor5" } } },
    }),
    { kind: "Ctor5" }
  );
}
assertions();
