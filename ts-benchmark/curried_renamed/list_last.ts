declare var require: any;
const assert = require("node:assert");
type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4"; f0: _uniq_0 };
type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_0; f1: _uniq_1 };
type _uniq_2 = { kind: "Ctor7" } | { kind: "Ctor8"; f0: _uniq_0 };

function _uniq_9(_uniq_10: _uniq_1): _uniq_2 {
  switch (_uniq_10.kind) {
    case "Ctor5": {
      return { kind: "Ctor7" };
    }
    case "Ctor6": {
      let _uniq_12 = _uniq_10.f1;
      let _uniq_11 = _uniq_10.f0;
      switch (_uniq_12.kind) {
        case "Ctor5": {
          return { kind: "Ctor8", f0: _uniq_11 };
        }
        case "Ctor6": {
          let _uniq_14 = _uniq_12.f1;
          let _uniq_13 = _uniq_12.f0;
          return _uniq_9(_uniq_12);
        }
      }
    }
  }
}

function assertions() {
  assert.deepEqual(_uniq_9({ kind: "Ctor5" }), { kind: "Ctor7" });
  assert.deepEqual(
    _uniq_9({
      kind: "Ctor6",
      f0: { kind: "Ctor4", f0: { kind: "Ctor3" } },
      f1: { kind: "Ctor5" },
    }),
    { kind: "Ctor8", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } }
  );
  assert.deepEqual(
    _uniq_9({
      kind: "Ctor6",
      f0: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
      f1: { kind: "Ctor5" },
    }),
    {
      kind: "Ctor8",
      f0: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
    }
  );
  assert.deepEqual(
    _uniq_9({
      kind: "Ctor6",
      f0: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
      f1: {
        kind: "Ctor6",
        f0: { kind: "Ctor4", f0: { kind: "Ctor3" } },
        f1: { kind: "Ctor5" },
      },
    }),
    { kind: "Ctor8", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } }
  );
  assert.deepEqual(
    _uniq_9({
      kind: "Ctor6",
      f0: { kind: "Ctor4", f0: { kind: "Ctor3" } },
      f1: {
        kind: "Ctor6",
        f0: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
        f1: { kind: "Ctor5" },
      },
    }),
    {
      kind: "Ctor8",
      f0: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
    }
  );
  assert.deepEqual(
    _uniq_9({
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
    { kind: "Ctor8", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } }
  );
}
assertions();

function validations() {
  assert.deepEqual(
    _uniq_9({
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
      kind: "Ctor8",
      f0: {
        kind: "Ctor4",
        f0: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
      },
    }
  );
  assert.deepEqual(
    _uniq_9({
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
      kind: "Ctor8",
      f0: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
    }
  );
  assert.deepEqual(
    _uniq_9({
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
    { kind: "Ctor8", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } }
  );
}
validations();
