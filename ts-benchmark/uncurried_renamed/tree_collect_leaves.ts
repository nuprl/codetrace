declare var require: any;
const assert = require("node:assert");
type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4" };
type _uniq_1 =
  | { kind: "Ctor5" }
  | { kind: "Ctor6"; f0: _uniq_1; f1: _uniq_0; f2: _uniq_1 };
type _uniq_2 = { kind: "Ctor7" } | { kind: "Ctor8"; f0: _uniq_0; f1: _uniq_2 };

function _uniq_9(_uniq_11: _uniq_2, _uniq_12: _uniq_2): _uniq_2 {
  switch (_uniq_11.kind) {
    case "Ctor7": {
      return _uniq_12;
    }
    case "Ctor8": {
      let _uniq_14 = _uniq_11.f1;
      let _uniq_13 = _uniq_11.f0;
      return { kind: "Ctor8", f0: _uniq_13, f1: _uniq_9(_uniq_14, _uniq_12) };
    }
  }
}

function _uniq_10(_uniq_15: _uniq_1): _uniq_2 {
  switch (_uniq_15.kind) {
    case "Ctor5": {
      return { kind: "Ctor7" };
    }
    case "Ctor6": {
      let _uniq_18 = _uniq_15.f2;
      let _uniq_17 = _uniq_15.f1;
      let _uniq_16 = _uniq_15.f0;
      switch (_uniq_16.kind) {
        case "Ctor5": {
          switch (_uniq_18.kind) {
            case "Ctor5": {
              return { kind: "Ctor8", f0: _uniq_17, f1: { kind: "Ctor7" } };
            }
            case "Ctor6": {
              let _uniq_21 = _uniq_18.f2;
              let _uniq_20 = _uniq_18.f1;
              let _uniq_19 = _uniq_18.f0;
              return { kind: "Ctor8", f0: _uniq_17, f1: _uniq_10(_uniq_18) };
            }
          }
        }
        case "Ctor6": {
          let _uniq_24 = _uniq_16.f2;
          let _uniq_23 = _uniq_16.f1;
          let _uniq_22 = _uniq_16.f0;
          switch (_uniq_18.kind) {
            case "Ctor5": {
              return _uniq_9(_uniq_10(_uniq_16), {
                kind: "Ctor8",
                f0: _uniq_17,
                f1: { kind: "Ctor7" },
              });
            }
            case "Ctor6": {
              let _uniq_27 = _uniq_18.f2;
              let _uniq_26 = _uniq_18.f1;
              let _uniq_25 = _uniq_18.f0;
              return _uniq_9(_uniq_10(_uniq_16), _uniq_10(_uniq_18));
            }
          }
        }
      }
    }
  }
}

function assertions() {
  assert.deepEqual(_uniq_10({ kind: "Ctor5" }), { kind: "Ctor7" });
  assert.deepEqual(
    _uniq_10({
      kind: "Ctor6",
      f0: {
        kind: "Ctor6",
        f0: { kind: "Ctor5" },
        f1: { kind: "Ctor3" },
        f2: { kind: "Ctor5" },
      },
      f1: { kind: "Ctor4" },
      f2: { kind: "Ctor5" },
    }),
    {
      kind: "Ctor8",
      f0: { kind: "Ctor3" },
      f1: { kind: "Ctor8", f0: { kind: "Ctor4" }, f1: { kind: "Ctor7" } },
    }
  );
  assert.deepEqual(
    _uniq_10({
      kind: "Ctor6",
      f0: { kind: "Ctor5" },
      f1: { kind: "Ctor4" },
      f2: {
        kind: "Ctor6",
        f0: { kind: "Ctor5" },
        f1: { kind: "Ctor3" },
        f2: { kind: "Ctor5" },
      },
    }),
    {
      kind: "Ctor8",
      f0: { kind: "Ctor4" },
      f1: { kind: "Ctor8", f0: { kind: "Ctor3" }, f1: { kind: "Ctor7" } },
    }
  );
}
assertions();

function validations() {
  assert.deepEqual(
    _uniq_10({
      kind: "Ctor6",
      f0: { kind: "Ctor5" },
      f1: { kind: "Ctor4" },
      f2: { kind: "Ctor5" },
    }),
    { kind: "Ctor8", f0: { kind: "Ctor4" }, f1: { kind: "Ctor7" } }
  );
  assert.deepEqual(
    _uniq_10({
      kind: "Ctor6",
      f0: {
        kind: "Ctor6",
        f0: { kind: "Ctor5" },
        f1: { kind: "Ctor3" },
        f2: {
          kind: "Ctor6",
          f0: { kind: "Ctor5" },
          f1: { kind: "Ctor4" },
          f2: { kind: "Ctor5" },
        },
      },
      f1: { kind: "Ctor3" },
      f2: {
        kind: "Ctor6",
        f0: { kind: "Ctor5" },
        f1: { kind: "Ctor4" },
        f2: { kind: "Ctor5" },
      },
    }),
    {
      kind: "Ctor8",
      f0: { kind: "Ctor3" },
      f1: {
        kind: "Ctor8",
        f0: { kind: "Ctor4" },
        f1: { kind: "Ctor8", f0: { kind: "Ctor4" }, f1: { kind: "Ctor7" } },
      },
    }
  );
  assert.deepEqual(
    _uniq_10({
      kind: "Ctor6",
      f0: {
        kind: "Ctor6",
        f0: {
          kind: "Ctor6",
          f0: { kind: "Ctor5" },
          f1: { kind: "Ctor3" },
          f2: { kind: "Ctor5" },
        },
        f1: { kind: "Ctor4" },
        f2: {
          kind: "Ctor6",
          f0: {
            kind: "Ctor6",
            f0: { kind: "Ctor5" },
            f1: { kind: "Ctor4" },
            f2: { kind: "Ctor5" },
          },
          f1: { kind: "Ctor3" },
          f2: { kind: "Ctor5" },
        },
      },
      f1: { kind: "Ctor3" },
      f2: {
        kind: "Ctor6",
        f0: { kind: "Ctor5" },
        f1: { kind: "Ctor4" },
        f2: {
          kind: "Ctor6",
          f0: { kind: "Ctor5" },
          f1: { kind: "Ctor3" },
          f2: { kind: "Ctor5" },
        },
      },
    }),
    {
      kind: "Ctor8",
      f0: { kind: "Ctor3" },
      f1: {
        kind: "Ctor8",
        f0: { kind: "Ctor4" },
        f1: {
          kind: "Ctor8",
          f0: { kind: "Ctor3" },
          f1: {
            kind: "Ctor8",
            f0: { kind: "Ctor4" },
            f1: { kind: "Ctor8", f0: { kind: "Ctor3" }, f1: { kind: "Ctor7" } },
          },
        },
      },
    }
  );
}
validations();
