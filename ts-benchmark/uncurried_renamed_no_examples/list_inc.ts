declare var require: any;
const assert = require("node:assert");
type _uniq_0 = { kind: "Ctor2" } | { kind: "Ctor3"; f0: _uniq_0 };
type _uniq_1 = { kind: "Ctor4" } | { kind: "Ctor5"; f0: _uniq_0; f1: _uniq_1 };

function _uniq_6(
  _uniq_8: _uniq_1,
  _uniq_9: (__x3: _uniq_0) => _uniq_0
): _uniq_1 {
  switch (_uniq_8.kind) {
    case "Ctor4": {
      return { kind: "Ctor4" };
    }
    case "Ctor5": {
      let _uniq_11 = _uniq_8.f1;
      let _uniq_10 = _uniq_8.f0;
      return {
        kind: "Ctor5",
        f0: _uniq_9(_uniq_10),
        f1: _uniq_6(_uniq_11, _uniq_9),
      };
    }
  }
}

function _uniq_7(_uniq_12: _uniq_1): _uniq_1 {
  return _uniq_6(_uniq_12, function (_uniq_13: _uniq_0) {
    return { kind: "Ctor3", f0: _uniq_13 };
  });
}

function assertions() {
  assert.deepEqual(_uniq_7({ kind: "Ctor4" }), { kind: "Ctor4" });
  assert.deepEqual(
    _uniq_7({
      kind: "Ctor5",
      f0: { kind: "Ctor3", f0: { kind: "Ctor2" } },
      f1: { kind: "Ctor4" },
    }),
    {
      kind: "Ctor5",
      f0: { kind: "Ctor3", f0: { kind: "Ctor3", f0: { kind: "Ctor2" } } },
      f1: { kind: "Ctor4" },
    }
  );
  assert.deepEqual(
    _uniq_7({
      kind: "Ctor5",
      f0: { kind: "Ctor3", f0: { kind: "Ctor2" } },
      f1: {
        kind: "Ctor5",
        f0: { kind: "Ctor3", f0: { kind: "Ctor3", f0: { kind: "Ctor2" } } },
        f1: { kind: "Ctor4" },
      },
    }),
    {
      kind: "Ctor5",
      f0: { kind: "Ctor3", f0: { kind: "Ctor3", f0: { kind: "Ctor2" } } },
      f1: {
        kind: "Ctor5",
        f0: {
          kind: "Ctor3",
          f0: { kind: "Ctor3", f0: { kind: "Ctor3", f0: { kind: "Ctor2" } } },
        },
        f1: { kind: "Ctor4" },
      },
    }
  );
  assert.deepEqual(
    _uniq_7({
      kind: "Ctor5",
      f0: { kind: "Ctor2" },
      f1: { kind: "Ctor5", f0: { kind: "Ctor2" }, f1: { kind: "Ctor4" } },
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
    _uniq_7({
      kind: "Ctor5",
      f0: {
        kind: "Ctor3",
        f0: { kind: "Ctor3", f0: { kind: "Ctor3", f0: { kind: "Ctor2" } } },
      },
      f1: {
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
          f1: { kind: "Ctor4" },
        },
      },
    }),
    {
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
      },
    }
  );
}
assertions();

function validations() {
  assert.deepEqual(
    _uniq_7({
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
      f0: {
        kind: "Ctor3",
        f0: {
          kind: "Ctor3",
          f0: {
            kind: "Ctor3",
            f0: { kind: "Ctor3", f0: { kind: "Ctor3", f0: { kind: "Ctor2" } } },
          },
        },
      },
      f1: {
        kind: "Ctor5",
        f0: { kind: "Ctor3", f0: { kind: "Ctor3", f0: { kind: "Ctor2" } } },
        f1: {
          kind: "Ctor5",
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
          f1: { kind: "Ctor4" },
        },
      },
    }
  );
  assert.deepEqual(
    _uniq_7({
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
      f0: { kind: "Ctor3", f0: { kind: "Ctor3", f0: { kind: "Ctor2" } } },
      f1: {
        kind: "Ctor5",
        f0: { kind: "Ctor3", f0: { kind: "Ctor3", f0: { kind: "Ctor2" } } },
        f1: {
          kind: "Ctor5",
          f0: { kind: "Ctor3", f0: { kind: "Ctor3", f0: { kind: "Ctor2" } } },
          f1: {
            kind: "Ctor5",
            f0: { kind: "Ctor3", f0: { kind: "Ctor3", f0: { kind: "Ctor2" } } },
            f1: { kind: "Ctor4" },
          },
        },
      },
    }
  );
  assert.deepEqual(
    _uniq_7({
      kind: "Ctor5",
      f0: { kind: "Ctor2" },
      f1: {
        kind: "Ctor5",
        f0: { kind: "Ctor3", f0: { kind: "Ctor2" } },
        f1: {
          kind: "Ctor5",
          f0: { kind: "Ctor3", f0: { kind: "Ctor3", f0: { kind: "Ctor2" } } },
          f1: {
            kind: "Ctor5",
            f0: {
              kind: "Ctor3",
              f0: {
                kind: "Ctor3",
                f0: { kind: "Ctor3", f0: { kind: "Ctor2" } },
              },
            },
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
        f0: { kind: "Ctor3", f0: { kind: "Ctor3", f0: { kind: "Ctor2" } } },
        f1: {
          kind: "Ctor5",
          f0: {
            kind: "Ctor3",
            f0: { kind: "Ctor3", f0: { kind: "Ctor3", f0: { kind: "Ctor2" } } },
          },
          f1: {
            kind: "Ctor5",
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
            f1: { kind: "Ctor4" },
          },
        },
      },
    }
  );
}
validations();
