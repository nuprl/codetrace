declare var require: any;
const assert = require("node:assert");
type _uniq_0 = { kind: "Ctor2" } | { kind: "Ctor3"; f0: _uniq_0 };
type _uniq_1 = { kind: "Ctor4" } | { kind: "Ctor5" } | { kind: "Ctor6" };

function _uniq_7(_uniq_9: _uniq_0): (_uniq_10: _uniq_0) => _uniq_1 {
  return function (_uniq_10: _uniq_0) {
    switch (_uniq_9.kind) {
      case "Ctor2": {
        switch (_uniq_10.kind) {
          case "Ctor2": {
            return { kind: "Ctor5" };
          }
          case "Ctor3": {
            let _uniq_11 = _uniq_10.f0;
            return { kind: "Ctor4" };
          }
        }
      }
      case "Ctor3": {
        let _uniq_12 = _uniq_9.f0;
        switch (_uniq_10.kind) {
          case "Ctor2": {
            return { kind: "Ctor6" };
          }
          case "Ctor3": {
            let _uniq_13 = _uniq_10.f0;
            return _uniq_7(_uniq_12)(_uniq_13);
          }
        }
      }
    }
  };
}

function _uniq_8(_uniq_14: _uniq_0): (_uniq_15: _uniq_0) => _uniq_0 {
  return function (_uniq_15: _uniq_0) {
    switch (_uniq_14.kind) {
      case "Ctor2": {
        return _uniq_15;
      }
      case "Ctor3": {
        let _uniq_16 = _uniq_14.f0;
        switch (_uniq_15.kind) {
          case "Ctor2": {
            return _uniq_14;
          }
          case "Ctor3": {
            let _uniq_17 = _uniq_15.f0;
            return { kind: "Ctor3", f0: _uniq_8(_uniq_16)(_uniq_17) };
          }
        }
      }
    }
  };
}

function assertions() {
  assert.deepEqual(_uniq_8({ kind: "Ctor2" })({ kind: "Ctor2" }), {
    kind: "Ctor2",
  });
  assert.deepEqual(
    _uniq_8({ kind: "Ctor2" })({ kind: "Ctor3", f0: { kind: "Ctor2" } }),
    { kind: "Ctor3", f0: { kind: "Ctor2" } }
  );
  assert.deepEqual(
    _uniq_8({ kind: "Ctor2" })({
      kind: "Ctor3",
      f0: { kind: "Ctor3", f0: { kind: "Ctor2" } },
    }),
    { kind: "Ctor3", f0: { kind: "Ctor3", f0: { kind: "Ctor2" } } }
  );
  assert.deepEqual(
    _uniq_8({ kind: "Ctor3", f0: { kind: "Ctor2" } })({ kind: "Ctor2" }),
    { kind: "Ctor3", f0: { kind: "Ctor2" } }
  );
  assert.deepEqual(
    _uniq_8({ kind: "Ctor3", f0: { kind: "Ctor2" } })({
      kind: "Ctor3",
      f0: { kind: "Ctor2" },
    }),
    { kind: "Ctor3", f0: { kind: "Ctor2" } }
  );
  assert.deepEqual(
    _uniq_8({ kind: "Ctor3", f0: { kind: "Ctor2" } })({
      kind: "Ctor3",
      f0: { kind: "Ctor3", f0: { kind: "Ctor2" } },
    }),
    { kind: "Ctor3", f0: { kind: "Ctor3", f0: { kind: "Ctor2" } } }
  );
  assert.deepEqual(
    _uniq_8({ kind: "Ctor3", f0: { kind: "Ctor3", f0: { kind: "Ctor2" } } })({
      kind: "Ctor2",
    }),
    { kind: "Ctor3", f0: { kind: "Ctor3", f0: { kind: "Ctor2" } } }
  );
  assert.deepEqual(
    _uniq_8({ kind: "Ctor3", f0: { kind: "Ctor3", f0: { kind: "Ctor2" } } })({
      kind: "Ctor3",
      f0: { kind: "Ctor2" },
    }),
    { kind: "Ctor3", f0: { kind: "Ctor3", f0: { kind: "Ctor2" } } }
  );
  assert.deepEqual(
    _uniq_8({ kind: "Ctor3", f0: { kind: "Ctor3", f0: { kind: "Ctor2" } } })({
      kind: "Ctor3",
      f0: { kind: "Ctor3", f0: { kind: "Ctor2" } },
    }),
    { kind: "Ctor3", f0: { kind: "Ctor3", f0: { kind: "Ctor2" } } }
  );
}
assertions();

function validations() {
  assert.deepEqual(
    _uniq_8({ kind: "Ctor3", f0: { kind: "Ctor3", f0: { kind: "Ctor2" } } })({
      kind: "Ctor3",
      f0: { kind: "Ctor3", f0: { kind: "Ctor3", f0: { kind: "Ctor2" } } },
    }),
    {
      kind: "Ctor3",
      f0: { kind: "Ctor3", f0: { kind: "Ctor3", f0: { kind: "Ctor2" } } },
    }
  );
  assert.deepEqual(
    _uniq_8({
      kind: "Ctor3",
      f0: { kind: "Ctor3", f0: { kind: "Ctor3", f0: { kind: "Ctor2" } } },
    })({ kind: "Ctor3", f0: { kind: "Ctor3", f0: { kind: "Ctor2" } } }),
    {
      kind: "Ctor3",
      f0: { kind: "Ctor3", f0: { kind: "Ctor3", f0: { kind: "Ctor2" } } },
    }
  );
  assert.deepEqual(
    _uniq_8({ kind: "Ctor3", f0: { kind: "Ctor3", f0: { kind: "Ctor2" } } })({
      kind: "Ctor3",
      f0: {
        kind: "Ctor3",
        f0: { kind: "Ctor3", f0: { kind: "Ctor3", f0: { kind: "Ctor2" } } },
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
  assert.deepEqual(
    _uniq_8({
      kind: "Ctor3",
      f0: {
        kind: "Ctor3",
        f0: { kind: "Ctor3", f0: { kind: "Ctor3", f0: { kind: "Ctor2" } } },
      },
    })({ kind: "Ctor2" }),
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
