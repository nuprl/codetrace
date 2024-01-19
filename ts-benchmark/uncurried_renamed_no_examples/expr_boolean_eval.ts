declare var require: any;
const assert = require("node:assert");
type _uniq_0 = { kind: "Ctor2" } | { kind: "Ctor3" };
type _uniq_1 =
  | { kind: "Ctor4"; f0: _uniq_0 }
  | { kind: "Ctor5"; f0: _uniq_1 }
  | { kind: "Ctor6"; f0: _uniq_1; f1: _uniq_1 }
  | { kind: "Ctor7"; f0: _uniq_1; f1: _uniq_1 }
  | { kind: "Ctor8"; f0: _uniq_1; f1: _uniq_1 };

function _uniq_9(_uniq_13: _uniq_0): _uniq_0 {
  switch (_uniq_13.kind) {
    case "Ctor2": {
      return { kind: "Ctor3" };
    }
    case "Ctor3": {
      return { kind: "Ctor2" };
    }
  }
}
function _uniq_10(_uniq_14: _uniq_0, _uniq_15: _uniq_0): _uniq_0 {
  switch (_uniq_15.kind) {
    case "Ctor2": {
      return _uniq_14;
    }
    case "Ctor3": {
      return { kind: "Ctor3" };
    }
  }
}
function _uniq_11(_uniq_16: _uniq_0, _uniq_17: _uniq_0): _uniq_0 {
  switch (_uniq_16.kind) {
    case "Ctor2": {
      return { kind: "Ctor2" };
    }
    case "Ctor3": {
      return _uniq_17;
    }
  }
}

function _uniq_12(_uniq_18: _uniq_1): _uniq_0 {
  switch (_uniq_18.kind) {
    case "Ctor4": {
      let _uniq_19 = _uniq_18.f0;
      return _uniq_19;
    }
    case "Ctor5": {
      let _uniq_20 = _uniq_18.f0;
      return _uniq_9(_uniq_12(_uniq_20));
    }
    case "Ctor6": {
      let _uniq_22 = _uniq_18.f1;
      let _uniq_21 = _uniq_18.f0;
      return _uniq_10(_uniq_12(_uniq_21), _uniq_12(_uniq_22));
    }
    case "Ctor7": {
      let _uniq_24 = _uniq_18.f1;
      let _uniq_23 = _uniq_18.f0;
      return _uniq_11(_uniq_12(_uniq_23), _uniq_12(_uniq_24));
    }
    case "Ctor8": {
      let _uniq_26 = _uniq_18.f1;
      let _uniq_25 = _uniq_18.f0;
      return _uniq_11(_uniq_9(_uniq_12(_uniq_25)), _uniq_12(_uniq_26));
    }
  }
}

function assertions() {
  assert.deepEqual(_uniq_12({ kind: "Ctor4", f0: { kind: "Ctor2" } }), {
    kind: "Ctor2",
  });
  assert.deepEqual(_uniq_12({ kind: "Ctor4", f0: { kind: "Ctor3" } }), {
    kind: "Ctor3",
  });
  assert.deepEqual(
    _uniq_12({ kind: "Ctor5", f0: { kind: "Ctor4", f0: { kind: "Ctor2" } } }),
    { kind: "Ctor3" }
  );
  assert.deepEqual(
    _uniq_12({ kind: "Ctor5", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } }),
    { kind: "Ctor2" }
  );
  assert.deepEqual(
    _uniq_12({
      kind: "Ctor6",
      f0: { kind: "Ctor4", f0: { kind: "Ctor2" } },
      f1: { kind: "Ctor4", f0: { kind: "Ctor3" } },
    }),
    { kind: "Ctor3" }
  );
  assert.deepEqual(
    _uniq_12({
      kind: "Ctor6",
      f0: { kind: "Ctor4", f0: { kind: "Ctor2" } },
      f1: { kind: "Ctor4", f0: { kind: "Ctor2" } },
    }),
    { kind: "Ctor2" }
  );
  assert.deepEqual(
    _uniq_12({
      kind: "Ctor6",
      f0: { kind: "Ctor4", f0: { kind: "Ctor3" } },
      f1: { kind: "Ctor4", f0: { kind: "Ctor2" } },
    }),
    { kind: "Ctor3" }
  );
  assert.deepEqual(
    _uniq_12({
      kind: "Ctor7",
      f0: { kind: "Ctor4", f0: { kind: "Ctor2" } },
      f1: { kind: "Ctor4", f0: { kind: "Ctor3" } },
    }),
    { kind: "Ctor2" }
  );
  assert.deepEqual(
    _uniq_12({
      kind: "Ctor7",
      f0: { kind: "Ctor4", f0: { kind: "Ctor3" } },
      f1: { kind: "Ctor4", f0: { kind: "Ctor3" } },
    }),
    { kind: "Ctor3" }
  );
  assert.deepEqual(
    _uniq_12({
      kind: "Ctor7",
      f0: { kind: "Ctor4", f0: { kind: "Ctor3" } },
      f1: { kind: "Ctor4", f0: { kind: "Ctor2" } },
    }),
    { kind: "Ctor2" }
  );
  assert.deepEqual(
    _uniq_12({
      kind: "Ctor8",
      f0: { kind: "Ctor4", f0: { kind: "Ctor2" } },
      f1: { kind: "Ctor4", f0: { kind: "Ctor3" } },
    }),
    { kind: "Ctor3" }
  );
  assert.deepEqual(
    _uniq_12({
      kind: "Ctor8",
      f0: { kind: "Ctor4", f0: { kind: "Ctor3" } },
      f1: { kind: "Ctor4", f0: { kind: "Ctor3" } },
    }),
    { kind: "Ctor2" }
  );
  assert.deepEqual(
    _uniq_12({
      kind: "Ctor8",
      f0: { kind: "Ctor4", f0: { kind: "Ctor3" } },
      f1: { kind: "Ctor4", f0: { kind: "Ctor2" } },
    }),
    { kind: "Ctor2" }
  );
  assert.deepEqual(
    _uniq_12({
      kind: "Ctor8",
      f0: { kind: "Ctor4", f0: { kind: "Ctor2" } },
      f1: { kind: "Ctor4", f0: { kind: "Ctor2" } },
    }),
    { kind: "Ctor2" }
  );
}
assertions();

function validations() {
  assert.deepEqual(
    _uniq_12({
      kind: "Ctor8",
      f0: { kind: "Ctor4", f0: { kind: "Ctor2" } },
      f1: {
        kind: "Ctor6",
        f0: { kind: "Ctor4", f0: { kind: "Ctor2" } },
        f1: { kind: "Ctor4", f0: { kind: "Ctor3" } },
      },
    }),
    { kind: "Ctor3" }
  );
  assert.deepEqual(
    _uniq_12({
      kind: "Ctor8",
      f0: {
        kind: "Ctor7",
        f0: { kind: "Ctor4", f0: { kind: "Ctor2" } },
        f1: { kind: "Ctor4", f0: { kind: "Ctor3" } },
      },
      f1: { kind: "Ctor4", f0: { kind: "Ctor2" } },
    }),
    { kind: "Ctor2" }
  );
  assert.deepEqual(
    _uniq_12({
      kind: "Ctor8",
      f0: {
        kind: "Ctor7",
        f0: { kind: "Ctor4", f0: { kind: "Ctor2" } },
        f1: { kind: "Ctor4", f0: { kind: "Ctor3" } },
      },
      f1: {
        kind: "Ctor7",
        f0: { kind: "Ctor4", f0: { kind: "Ctor2" } },
        f1: { kind: "Ctor4", f0: { kind: "Ctor3" } },
      },
    }),
    { kind: "Ctor2" }
  );
}
validations();
