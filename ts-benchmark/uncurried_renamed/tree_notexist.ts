declare var require: any;
const assert = require("node:assert");
type _uniq_0 = { kind: "Ctor4" } | { kind: "Ctor5"; f0: _uniq_0 };
type _uniq_1 = { kind: "Ctor6" } | { kind: "Ctor7" };
type _uniq_2 = { kind: "Ctor8" } | { kind: "Ctor9" } | { kind: "Ctor10" };
type _uniq_3 =
  | { kind: "Ctor11" }
  | { kind: "Ctor12"; f0: _uniq_0; f1: _uniq_3; f2: _uniq_3 };

function _uniq_13(_uniq_16: _uniq_0, _uniq_17: _uniq_0): _uniq_2 {
  switch (_uniq_16.kind) {
    case "Ctor4": {
      switch (_uniq_17.kind) {
        case "Ctor4": {
          return { kind: "Ctor9" };
        }
        case "Ctor5": {
          let _uniq_18 = _uniq_17.f0;
          return { kind: "Ctor8" };
        }
      }
    }
    case "Ctor5": {
      let _uniq_19 = _uniq_16.f0;
      switch (_uniq_17.kind) {
        case "Ctor4": {
          return { kind: "Ctor10" };
        }
        case "Ctor5": {
          let _uniq_20 = _uniq_17.f0;
          return _uniq_13(_uniq_19, _uniq_20);
        }
      }
    }
  }
}
function _uniq_14(_uniq_21: _uniq_1, _uniq_22: _uniq_1): _uniq_1 {
  switch (_uniq_21.kind) {
    case "Ctor6": {
      return _uniq_22;
    }
    case "Ctor7": {
      return { kind: "Ctor7" };
    }
  }
}

function _uniq_15(_uniq_23: _uniq_0, _uniq_24: _uniq_3): _uniq_1 {
  switch (_uniq_24.kind) {
    case "Ctor11": {
      return { kind: "Ctor6" };
    }
    case "Ctor12": {
      let _uniq_27 = _uniq_24.f2;
      let _uniq_26 = _uniq_24.f1;
      let _uniq_25 = _uniq_24.f0;
      switch (_uniq_13(_uniq_25, _uniq_23).kind) {
        case "Ctor9": {
          return { kind: "Ctor7" };
        }
        case "Ctor8": {
          return _uniq_14(
            _uniq_15(_uniq_23, _uniq_26),
            _uniq_15(_uniq_23, _uniq_27)
          );
        }
        case "Ctor10": {
          return _uniq_14(
            _uniq_15(_uniq_23, _uniq_26),
            _uniq_15(_uniq_23, _uniq_27)
          );
        }
      }
    }
  }
}

function assertions() {
  assert.deepEqual(
    _uniq_15({ kind: "Ctor5", f0: { kind: "Ctor4" } }, { kind: "Ctor11" }),
    { kind: "Ctor6" }
  );
  assert.deepEqual(_uniq_15({ kind: "Ctor4" }, { kind: "Ctor11" }), {
    kind: "Ctor6",
  });
  assert.deepEqual(
    _uniq_15(
      { kind: "Ctor4" },
      {
        kind: "Ctor12",
        f0: { kind: "Ctor4" },
        f1: { kind: "Ctor11" },
        f2: { kind: "Ctor11" },
      }
    ),
    { kind: "Ctor7" }
  );
  assert.deepEqual(
    _uniq_15(
      { kind: "Ctor5", f0: { kind: "Ctor4" } },
      {
        kind: "Ctor12",
        f0: { kind: "Ctor4" },
        f1: { kind: "Ctor11" },
        f2: { kind: "Ctor11" },
      }
    ),
    { kind: "Ctor6" }
  );
  assert.deepEqual(
    _uniq_15(
      { kind: "Ctor4" },
      {
        kind: "Ctor12",
        f0: { kind: "Ctor5", f0: { kind: "Ctor4" } },
        f1: { kind: "Ctor11" },
        f2: { kind: "Ctor11" },
      }
    ),
    { kind: "Ctor6" }
  );
  assert.deepEqual(
    _uniq_15(
      { kind: "Ctor4" },
      {
        kind: "Ctor12",
        f0: { kind: "Ctor5", f0: { kind: "Ctor4" } },
        f1: {
          kind: "Ctor12",
          f0: { kind: "Ctor4" },
          f1: { kind: "Ctor11" },
          f2: { kind: "Ctor11" },
        },
        f2: { kind: "Ctor11" },
      }
    ),
    { kind: "Ctor7" }
  );
  assert.deepEqual(
    _uniq_15(
      { kind: "Ctor5", f0: { kind: "Ctor4" } },
      {
        kind: "Ctor12",
        f0: { kind: "Ctor4" },
        f1: {
          kind: "Ctor12",
          f0: { kind: "Ctor5", f0: { kind: "Ctor4" } },
          f1: { kind: "Ctor11" },
          f2: { kind: "Ctor11" },
        },
        f2: { kind: "Ctor11" },
      }
    ),
    { kind: "Ctor7" }
  );
  assert.deepEqual(
    _uniq_15(
      { kind: "Ctor4" },
      {
        kind: "Ctor12",
        f0: { kind: "Ctor5", f0: { kind: "Ctor4" } },
        f1: { kind: "Ctor11" },
        f2: {
          kind: "Ctor12",
          f0: { kind: "Ctor4" },
          f1: { kind: "Ctor11" },
          f2: { kind: "Ctor11" },
        },
      }
    ),
    { kind: "Ctor7" }
  );
  assert.deepEqual(
    _uniq_15(
      { kind: "Ctor5", f0: { kind: "Ctor4" } },
      {
        kind: "Ctor12",
        f0: { kind: "Ctor4" },
        f1: { kind: "Ctor11" },
        f2: {
          kind: "Ctor12",
          f0: { kind: "Ctor4" },
          f1: { kind: "Ctor11" },
          f2: { kind: "Ctor11" },
        },
      }
    ),
    { kind: "Ctor6" }
  );
  assert.deepEqual(
    _uniq_15(
      { kind: "Ctor5", f0: { kind: "Ctor4" } },
      {
        kind: "Ctor12",
        f0: { kind: "Ctor4" },
        f1: { kind: "Ctor11" },
        f2: {
          kind: "Ctor12",
          f0: { kind: "Ctor5", f0: { kind: "Ctor4" } },
          f1: { kind: "Ctor11" },
          f2: { kind: "Ctor11" },
        },
      }
    ),
    { kind: "Ctor7" }
  );
  assert.deepEqual(
    _uniq_15(
      { kind: "Ctor5", f0: { kind: "Ctor5", f0: { kind: "Ctor4" } } },
      {
        kind: "Ctor12",
        f0: { kind: "Ctor4" },
        f1: {
          kind: "Ctor12",
          f0: { kind: "Ctor5", f0: { kind: "Ctor4" } },
          f1: { kind: "Ctor11" },
          f2: { kind: "Ctor11" },
        },
        f2: {
          kind: "Ctor12",
          f0: { kind: "Ctor5", f0: { kind: "Ctor5", f0: { kind: "Ctor4" } } },
          f1: { kind: "Ctor11" },
          f2: { kind: "Ctor11" },
        },
      }
    ),
    { kind: "Ctor7" }
  );
  assert.deepEqual(
    _uniq_15(
      {
        kind: "Ctor5",
        f0: { kind: "Ctor5", f0: { kind: "Ctor5", f0: { kind: "Ctor4" } } },
      },
      {
        kind: "Ctor12",
        f0: { kind: "Ctor4" },
        f1: {
          kind: "Ctor12",
          f0: { kind: "Ctor5", f0: { kind: "Ctor4" } },
          f1: {
            kind: "Ctor12",
            f0: { kind: "Ctor5", f0: { kind: "Ctor5", f0: { kind: "Ctor4" } } },
            f1: { kind: "Ctor11" },
            f2: { kind: "Ctor11" },
          },
          f2: { kind: "Ctor11" },
        },
        f2: { kind: "Ctor11" },
      }
    ),
    { kind: "Ctor6" }
  );
}
assertions();
