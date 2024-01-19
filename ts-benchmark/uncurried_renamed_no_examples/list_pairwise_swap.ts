declare var require: any;
const assert = require("node:assert");
type _uniq_0 = { kind: "Ctor2" } | { kind: "Ctor3"; f0: _uniq_0 };
type _uniq_1 = { kind: "Ctor4" } | { kind: "Ctor5"; f0: _uniq_0; f1: _uniq_1 };

function _uniq_6(_uniq_7: _uniq_1): _uniq_1 {
  switch (_uniq_7.kind) {
    case "Ctor4": {
      return { kind: "Ctor4" };
    }
    case "Ctor5": {
      let _uniq_9 = _uniq_7.f1;
      let _uniq_8 = _uniq_7.f0;
      switch (_uniq_9.kind) {
        case "Ctor4": {
          return { kind: "Ctor4" };
        }
        case "Ctor5": {
          let _uniq_11 = _uniq_9.f1;
          let _uniq_10 = _uniq_9.f0;
          return {
            kind: "Ctor5",
            f0: _uniq_10,
            f1: { kind: "Ctor5", f0: _uniq_8, f1: _uniq_6(_uniq_11) },
          };
        }
      }
    }
  }
}

function assertions() {
  assert.deepEqual(_uniq_6({ kind: "Ctor4" }), { kind: "Ctor4" });
  assert.deepEqual(
    _uniq_6({
      kind: "Ctor5",
      f0: { kind: "Ctor3", f0: { kind: "Ctor2" } },
      f1: { kind: "Ctor4" },
    }),
    { kind: "Ctor4" }
  );
  assert.deepEqual(
    _uniq_6({
      kind: "Ctor5",
      f0: { kind: "Ctor3", f0: { kind: "Ctor2" } },
      f1: { kind: "Ctor5", f0: { kind: "Ctor2" }, f1: { kind: "Ctor4" } },
    }),
    {
      kind: "Ctor5",
      f0: { kind: "Ctor2" },
      f1: {
        kind: "Ctor5",
        f0: { kind: "Ctor3", f0: { kind: "Ctor2" } },
        f1: { kind: "Ctor4" },
      },
    }
  );
  assert.deepEqual(
    _uniq_6({
      kind: "Ctor5",
      f0: { kind: "Ctor3", f0: { kind: "Ctor2" } },
      f1: {
        kind: "Ctor5",
        f0: { kind: "Ctor2" },
        f1: {
          kind: "Ctor5",
          f0: { kind: "Ctor3", f0: { kind: "Ctor2" } },
          f1: { kind: "Ctor4" },
        },
      },
    }),
    {
      kind: "Ctor5",
      f0: { kind: "Ctor2" },
      f1: {
        kind: "Ctor5",
        f0: { kind: "Ctor3", f0: { kind: "Ctor2" } },
        f1: { kind: "Ctor4" },
      },
    }
  );
  assert.deepEqual(
    _uniq_6({
      kind: "Ctor5",
      f0: { kind: "Ctor2" },
      f1: {
        kind: "Ctor5",
        f0: { kind: "Ctor3", f0: { kind: "Ctor2" } },
        f1: {
          kind: "Ctor5",
          f0: { kind: "Ctor2" },
          f1: {
            kind: "Ctor5",
            f0: { kind: "Ctor3", f0: { kind: "Ctor2" } },
            f1: { kind: "Ctor4" },
          },
        },
      },
    }),
    {
      kind: "Ctor5",
      f0: { kind: "Ctor3", f0: { kind: "Ctor2" } },
      f1: {
        kind: "Ctor5",
        f0: { kind: "Ctor2" },
        f1: {
          kind: "Ctor5",
          f0: { kind: "Ctor3", f0: { kind: "Ctor2" } },
          f1: { kind: "Ctor5", f0: { kind: "Ctor2" }, f1: { kind: "Ctor4" } },
        },
      },
    }
  );
}
assertions();

function validations() {
  assert.deepEqual(
    _uniq_6({
      kind: "Ctor5",
      f0: {
        kind: "Ctor3",
        f0: {
          kind: "Ctor3",
          f0: { kind: "Ctor3", f0: { kind: "Ctor3", f0: { kind: "Ctor2" } } },
        },
      },
      f1: {
        kind: "Ctor5",
        f0: { kind: "Ctor3", f0: { kind: "Ctor2" } },
        f1: {
          kind: "Ctor5",
          f0: {
            kind: "Ctor3",
            f0: { kind: "Ctor3", f0: { kind: "Ctor3", f0: { kind: "Ctor2" } } },
          },
          f1: { kind: "Ctor4" },
        },
      },
    }),
    {
      kind: "Ctor5",
      f0: { kind: "Ctor3", f0: { kind: "Ctor2" } },
      f1: {
        kind: "Ctor5",
        f0: {
          kind: "Ctor3",
          f0: {
            kind: "Ctor3",
            f0: { kind: "Ctor3", f0: { kind: "Ctor3", f0: { kind: "Ctor2" } } },
          },
        },
        f1: { kind: "Ctor4" },
      },
    }
  );
  assert.deepEqual(
    _uniq_6({
      kind: "Ctor5",
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
      f1: {
        kind: "Ctor5",
        f0: { kind: "Ctor3", f0: { kind: "Ctor2" } },
        f1: {
          kind: "Ctor5",
          f0: { kind: "Ctor3", f0: { kind: "Ctor3", f0: { kind: "Ctor2" } } },
          f1: { kind: "Ctor4" },
        },
      },
    }),
    {
      kind: "Ctor5",
      f0: { kind: "Ctor3", f0: { kind: "Ctor2" } },
      f1: {
        kind: "Ctor5",
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
        f1: { kind: "Ctor4" },
      },
    }
  );
  assert.deepEqual(
    _uniq_6({
      kind: "Ctor5",
      f0: { kind: "Ctor3", f0: { kind: "Ctor2" } },
      f1: {
        kind: "Ctor5",
        f0: { kind: "Ctor3", f0: { kind: "Ctor2" } },
        f1: {
          kind: "Ctor5",
          f0: { kind: "Ctor3", f0: { kind: "Ctor2" } },
          f1: { kind: "Ctor4" },
        },
      },
    }),
    {
      kind: "Ctor5",
      f0: { kind: "Ctor3", f0: { kind: "Ctor2" } },
      f1: {
        kind: "Ctor5",
        f0: { kind: "Ctor3", f0: { kind: "Ctor2" } },
        f1: { kind: "Ctor4" },
      },
    }
  );
  assert.deepEqual(
    _uniq_6({
      kind: "Ctor5",
      f0: { kind: "Ctor3", f0: { kind: "Ctor2" } },
      f1: {
        kind: "Ctor5",
        f0: { kind: "Ctor3", f0: { kind: "Ctor2" } },
        f1: {
          kind: "Ctor5",
          f0: { kind: "Ctor3", f0: { kind: "Ctor2" } },
          f1: {
            kind: "Ctor5",
            f0: { kind: "Ctor3", f0: { kind: "Ctor2" } },
            f1: { kind: "Ctor4" },
          },
        },
      },
    }),
    {
      kind: "Ctor5",
      f0: { kind: "Ctor3", f0: { kind: "Ctor2" } },
      f1: {
        kind: "Ctor5",
        f0: { kind: "Ctor3", f0: { kind: "Ctor2" } },
        f1: {
          kind: "Ctor5",
          f0: { kind: "Ctor3", f0: { kind: "Ctor2" } },
          f1: {
            kind: "Ctor5",
            f0: { kind: "Ctor3", f0: { kind: "Ctor2" } },
            f1: { kind: "Ctor4" },
          },
        },
      },
    }
  );
}
validations();
