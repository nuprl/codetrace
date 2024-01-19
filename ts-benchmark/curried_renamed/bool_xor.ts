declare var require: any;
const assert = require("node:assert");
type _uniq_0 = { kind: "Ctor1" } | { kind: "Ctor2" };

function _uniq_3(_uniq_4: _uniq_0): (_uniq_5: _uniq_0) => _uniq_0 {
  return function (_uniq_5: _uniq_0) {
    switch (_uniq_4.kind) {
      case "Ctor1": {
        switch (_uniq_5.kind) {
          case "Ctor2": {
            return { kind: "Ctor1" };
          }
          case "Ctor1": {
            return { kind: "Ctor2" };
          }
        }
      }
      case "Ctor2": {
        return _uniq_5;
      }
    }
  };
}

function assertions() {
  assert.deepEqual(_uniq_3({ kind: "Ctor1" })({ kind: "Ctor1" }), {
    kind: "Ctor2",
  });
  assert.deepEqual(_uniq_3({ kind: "Ctor1" })({ kind: "Ctor2" }), {
    kind: "Ctor1",
  });
  assert.deepEqual(_uniq_3({ kind: "Ctor2" })({ kind: "Ctor1" }), {
    kind: "Ctor1",
  });
  assert.deepEqual(_uniq_3({ kind: "Ctor2" })({ kind: "Ctor2" }), {
    kind: "Ctor2",
  });
}
assertions();

function validations() {
  assert.deepEqual(_uniq_3({ kind: "Ctor1" })({ kind: "Ctor1" }), {
    kind: "Ctor2",
  });
  assert.deepEqual(_uniq_3({ kind: "Ctor1" })({ kind: "Ctor2" }), {
    kind: "Ctor1",
  });
  assert.deepEqual(_uniq_3({ kind: "Ctor2" })({ kind: "Ctor1" }), {
    kind: "Ctor1",
  });
  assert.deepEqual(_uniq_3({ kind: "Ctor2" })({ kind: "Ctor2" }), {
    kind: "Ctor2",
  });
}
validations();
