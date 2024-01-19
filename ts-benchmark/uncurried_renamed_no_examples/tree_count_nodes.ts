declare var require: any;
const assert = require("node:assert");
type _uniq_0 = { kind: "Ctor2" } | { kind: "Ctor3"; f0: _uniq_0 };
type _uniq_1 =
  | { kind: "Ctor4" }
  | { kind: "Ctor5"; f0: _uniq_1; f1: _uniq_0; f2: _uniq_1 };

function _uniq_6(_uniq_8: _uniq_0, _uniq_9: _uniq_0): _uniq_0 {
  switch (_uniq_8.kind) {
    case "Ctor2": {
      return _uniq_9;
    }
    case "Ctor3": {
      let _uniq_10 = _uniq_8.f0;
      return { kind: "Ctor3", f0: _uniq_6(_uniq_10, _uniq_9) };
    }
  }
}

function _uniq_7(_uniq_11: _uniq_1): _uniq_0 {
  switch (_uniq_11.kind) {
    case "Ctor4": {
      return { kind: "Ctor2" };
    }
    case "Ctor5": {
      let _uniq_14 = _uniq_11.f2;
      let _uniq_13 = _uniq_11.f1;
      let _uniq_12 = _uniq_11.f0;
      return {
        kind: "Ctor3",
        f0: _uniq_6(_uniq_7(_uniq_12), _uniq_7(_uniq_14)),
      };
    }
  }
}

function assertions() {
  assert.deepEqual(_uniq_7({ kind: "Ctor4" }), { kind: "Ctor2" });
  assert.deepEqual(
    _uniq_7({
      kind: "Ctor5",
      f0: { kind: "Ctor4" },
      f1: { kind: "Ctor2" },
      f2: { kind: "Ctor4" },
    }),
    { kind: "Ctor3", f0: { kind: "Ctor2" } }
  );
  assert.deepEqual(
    _uniq_7({
      kind: "Ctor5",
      f0: {
        kind: "Ctor5",
        f0: { kind: "Ctor4" },
        f1: { kind: "Ctor2" },
        f2: { kind: "Ctor4" },
      },
      f1: { kind: "Ctor2" },
      f2: { kind: "Ctor4" },
    }),
    { kind: "Ctor3", f0: { kind: "Ctor3", f0: { kind: "Ctor2" } } }
  );
  assert.deepEqual(
    _uniq_7({
      kind: "Ctor5",
      f0: { kind: "Ctor4" },
      f1: { kind: "Ctor2" },
      f2: {
        kind: "Ctor5",
        f0: { kind: "Ctor4" },
        f1: { kind: "Ctor2" },
        f2: { kind: "Ctor4" },
      },
    }),
    { kind: "Ctor3", f0: { kind: "Ctor3", f0: { kind: "Ctor2" } } }
  );
  assert.deepEqual(
    _uniq_7({
      kind: "Ctor5",
      f0: {
        kind: "Ctor5",
        f0: { kind: "Ctor4" },
        f1: { kind: "Ctor2" },
        f2: {
          kind: "Ctor5",
          f0: { kind: "Ctor4" },
          f1: { kind: "Ctor2" },
          f2: { kind: "Ctor4" },
        },
      },
      f1: { kind: "Ctor2" },
      f2: { kind: "Ctor4" },
    }),
    {
      kind: "Ctor3",
      f0: { kind: "Ctor3", f0: { kind: "Ctor3", f0: { kind: "Ctor2" } } },
    }
  );
  assert.deepEqual(
    _uniq_7({
      kind: "Ctor5",
      f0: { kind: "Ctor4" },
      f1: { kind: "Ctor2" },
      f2: {
        kind: "Ctor5",
        f0: { kind: "Ctor4" },
        f1: { kind: "Ctor2" },
        f2: {
          kind: "Ctor5",
          f0: { kind: "Ctor4" },
          f1: { kind: "Ctor2" },
          f2: { kind: "Ctor4" },
        },
      },
    }),
    {
      kind: "Ctor3",
      f0: { kind: "Ctor3", f0: { kind: "Ctor3", f0: { kind: "Ctor2" } } },
    }
  );
}
assertions();

function validations() {
  assert.deepEqual(
    _uniq_7({
      kind: "Ctor5",
      f0: { kind: "Ctor4" },
      f1: { kind: "Ctor3", f0: { kind: "Ctor2" } },
      f2: {
        kind: "Ctor5",
        f0: {
          kind: "Ctor5",
          f0: { kind: "Ctor4" },
          f1: { kind: "Ctor3", f0: { kind: "Ctor3", f0: { kind: "Ctor2" } } },
          f2: { kind: "Ctor4" },
        },
        f1: { kind: "Ctor3", f0: { kind: "Ctor2" } },
        f2: {
          kind: "Ctor5",
          f0: { kind: "Ctor4" },
          f1: { kind: "Ctor3", f0: { kind: "Ctor2" } },
          f2: { kind: "Ctor4" },
        },
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
    _uniq_7({
      kind: "Ctor5",
      f0: {
        kind: "Ctor5",
        f0: {
          kind: "Ctor5",
          f0: { kind: "Ctor4" },
          f1: { kind: "Ctor2" },
          f2: { kind: "Ctor4" },
        },
        f1: { kind: "Ctor3", f0: { kind: "Ctor2" } },
        f2: {
          kind: "Ctor5",
          f0: { kind: "Ctor4" },
          f1: { kind: "Ctor3", f0: { kind: "Ctor2" } },
          f2: { kind: "Ctor4" },
        },
      },
      f1: { kind: "Ctor3", f0: { kind: "Ctor3", f0: { kind: "Ctor2" } } },
      f2: {
        kind: "Ctor5",
        f0: {
          kind: "Ctor5",
          f0: { kind: "Ctor4" },
          f1: { kind: "Ctor3", f0: { kind: "Ctor2" } },
          f2: { kind: "Ctor4" },
        },
        f1: { kind: "Ctor2" },
        f2: {
          kind: "Ctor5",
          f0: { kind: "Ctor4" },
          f1: { kind: "Ctor2" },
          f2: { kind: "Ctor4" },
        },
      },
    }),
    {
      kind: "Ctor3",
      f0: {
        kind: "Ctor3",
        f0: {
          kind: "Ctor3",
          f0: {
            kind: "Ctor3",
            f0: {
              kind: "Ctor3",
              f0: {
                kind: "Ctor3",
                f0: { kind: "Ctor3", f0: { kind: "Ctor2" } },
              },
            },
          },
        },
      },
    }
  );
  assert.deepEqual(
    _uniq_7({
      kind: "Ctor5",
      f0: {
        kind: "Ctor5",
        f0: { kind: "Ctor4" },
        f1: { kind: "Ctor2" },
        f2: { kind: "Ctor4" },
      },
      f1: { kind: "Ctor3", f0: { kind: "Ctor2" } },
      f2: {
        kind: "Ctor5",
        f0: { kind: "Ctor4" },
        f1: { kind: "Ctor2" },
        f2: { kind: "Ctor4" },
      },
    }),
    {
      kind: "Ctor3",
      f0: { kind: "Ctor3", f0: { kind: "Ctor3", f0: { kind: "Ctor2" } } },
    }
  );
}
validations();
