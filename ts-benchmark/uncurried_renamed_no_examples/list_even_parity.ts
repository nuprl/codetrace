declare var require: any;
const assert = require("node:assert");
type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4"; f0: _uniq_0 };
type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6" };
type _uniq_2 = { kind: "Ctor7" } | { kind: "Ctor8"; f0: _uniq_1; f1: _uniq_2 };

function _uniq_9(_uniq_10: _uniq_2): _uniq_1 {
  switch (_uniq_10.kind) {
    case "Ctor7": {
      return { kind: "Ctor5" };
    }
    case "Ctor8": {
      let _uniq_12 = _uniq_10.f1;
      let _uniq_11 = _uniq_10.f0;
      let _uniq_13: _uniq_1 = _uniq_9(_uniq_12);
      switch (_uniq_11.kind) {
        case "Ctor5": {
          switch (_uniq_13.kind) {
            case "Ctor5": {
              return { kind: "Ctor6" };
            }
            case "Ctor6": {
              return { kind: "Ctor5" };
            }
          }
        }
        case "Ctor6": {
          return _uniq_13;
        }
      }
    }
  }
}

function assertions() {
  assert.deepEqual(_uniq_9({ kind: "Ctor7" }), { kind: "Ctor5" });
  assert.deepEqual(
    _uniq_9({ kind: "Ctor8", f0: { kind: "Ctor6" }, f1: { kind: "Ctor7" } }),
    { kind: "Ctor5" }
  );
  assert.deepEqual(
    _uniq_9({ kind: "Ctor8", f0: { kind: "Ctor5" }, f1: { kind: "Ctor7" } }),
    { kind: "Ctor6" }
  );
  assert.deepEqual(
    _uniq_9({
      kind: "Ctor8",
      f0: { kind: "Ctor6" },
      f1: { kind: "Ctor8", f0: { kind: "Ctor6" }, f1: { kind: "Ctor7" } },
    }),
    { kind: "Ctor5" }
  );
  assert.deepEqual(
    _uniq_9({
      kind: "Ctor8",
      f0: { kind: "Ctor6" },
      f1: { kind: "Ctor8", f0: { kind: "Ctor5" }, f1: { kind: "Ctor7" } },
    }),
    { kind: "Ctor6" }
  );
  assert.deepEqual(
    _uniq_9({
      kind: "Ctor8",
      f0: { kind: "Ctor5" },
      f1: { kind: "Ctor8", f0: { kind: "Ctor6" }, f1: { kind: "Ctor7" } },
    }),
    { kind: "Ctor6" }
  );
  assert.deepEqual(
    _uniq_9({
      kind: "Ctor8",
      f0: { kind: "Ctor5" },
      f1: { kind: "Ctor8", f0: { kind: "Ctor5" }, f1: { kind: "Ctor7" } },
    }),
    { kind: "Ctor5" }
  );
}
assertions();

function validations() {
  assert.deepEqual(
    _uniq_9({
      kind: "Ctor8",
      f0: { kind: "Ctor6" },
      f1: {
        kind: "Ctor8",
        f0: { kind: "Ctor6" },
        f1: { kind: "Ctor8", f0: { kind: "Ctor6" }, f1: { kind: "Ctor7" } },
      },
    }),
    { kind: "Ctor5" }
  );
  assert.deepEqual(
    _uniq_9({
      kind: "Ctor8",
      f0: { kind: "Ctor6" },
      f1: {
        kind: "Ctor8",
        f0: { kind: "Ctor5" },
        f1: { kind: "Ctor8", f0: { kind: "Ctor6" }, f1: { kind: "Ctor7" } },
      },
    }),
    { kind: "Ctor6" }
  );
  assert.deepEqual(
    _uniq_9({
      kind: "Ctor8",
      f0: { kind: "Ctor5" },
      f1: {
        kind: "Ctor8",
        f0: { kind: "Ctor6" },
        f1: { kind: "Ctor8", f0: { kind: "Ctor5" }, f1: { kind: "Ctor7" } },
      },
    }),
    { kind: "Ctor5" }
  );
  assert.deepEqual(
    _uniq_9({
      kind: "Ctor8",
      f0: { kind: "Ctor5" },
      f1: {
        kind: "Ctor8",
        f0: { kind: "Ctor5" },
        f1: { kind: "Ctor8", f0: { kind: "Ctor5" }, f1: { kind: "Ctor7" } },
      },
    }),
    { kind: "Ctor6" }
  );
}
validations();
