declare var require: any;
const assert = require("node:assert");
type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4" } | { kind: "Ctor5" };
type _uniq_1 = { kind: "Ctor6" } | { kind: "Ctor7"; f0: _uniq_1 };
type _uniq_2 =
  | { kind: "Ctor8" }
  | { kind: "Ctor9"; f0: _uniq_2; f1: _uniq_1; f2: _uniq_2 };

function _uniq_10(_uniq_12: _uniq_1, _uniq_13: _uniq_1): _uniq_0 {
  switch (_uniq_12.kind) {
    case "Ctor6": {
      switch (_uniq_13.kind) {
        case "Ctor6": {
          return { kind: "Ctor3" };
        }
        case "Ctor7": {
          let _uniq_14 = _uniq_13.f0;
          return { kind: "Ctor5" };
        }
      }
    }
    case "Ctor7": {
      let _uniq_15 = _uniq_12.f0;
      switch (_uniq_13.kind) {
        case "Ctor6": {
          return { kind: "Ctor4" };
        }
        case "Ctor7": {
          let _uniq_16 = _uniq_13.f0;
          return _uniq_10(_uniq_15, _uniq_16);
        }
      }
    }
  }
}

function _uniq_11(_uniq_17: _uniq_2, _uniq_18: _uniq_1): _uniq_2 {
  switch (_uniq_17.kind) {
    case "Ctor8": {
      return {
        kind: "Ctor9",
        f0: { kind: "Ctor8" },
        f1: _uniq_18,
        f2: { kind: "Ctor8" },
      };
    }
    case "Ctor9": {
      let _uniq_21 = _uniq_17.f2;
      let _uniq_20 = _uniq_17.f1;
      let _uniq_19 = _uniq_17.f0;
      switch (_uniq_10(_uniq_18, _uniq_20).kind) {
        case "Ctor3": {
          return { kind: "Ctor9", f0: _uniq_19, f1: _uniq_20, f2: _uniq_21 };
        }
        case "Ctor5": {
          return {
            kind: "Ctor9",
            f0: _uniq_11(_uniq_19, _uniq_18),
            f1: _uniq_20,
            f2: _uniq_21,
          };
        }
        case "Ctor4": {
          return {
            kind: "Ctor9",
            f0: _uniq_19,
            f1: _uniq_20,
            f2: _uniq_11(_uniq_21, _uniq_18),
          };
        }
      }
    }
  }
}

function assertions() {
  assert.deepEqual(_uniq_11({ kind: "Ctor8" }, { kind: "Ctor6" }), {
    kind: "Ctor9",
    f0: { kind: "Ctor8" },
    f1: { kind: "Ctor6" },
    f2: { kind: "Ctor8" },
  });
  assert.deepEqual(
    _uniq_11({ kind: "Ctor8" }, { kind: "Ctor7", f0: { kind: "Ctor6" } }),
    {
      kind: "Ctor9",
      f0: { kind: "Ctor8" },
      f1: { kind: "Ctor7", f0: { kind: "Ctor6" } },
      f2: { kind: "Ctor8" },
    }
  );
  assert.deepEqual(
    _uniq_11(
      { kind: "Ctor8" },
      { kind: "Ctor7", f0: { kind: "Ctor7", f0: { kind: "Ctor6" } } }
    ),
    {
      kind: "Ctor9",
      f0: { kind: "Ctor8" },
      f1: { kind: "Ctor7", f0: { kind: "Ctor7", f0: { kind: "Ctor6" } } },
      f2: { kind: "Ctor8" },
    }
  );
  assert.deepEqual(
    _uniq_11(
      {
        kind: "Ctor9",
        f0: { kind: "Ctor8" },
        f1: { kind: "Ctor6" },
        f2: { kind: "Ctor8" },
      },
      { kind: "Ctor6" }
    ),
    {
      kind: "Ctor9",
      f0: { kind: "Ctor8" },
      f1: { kind: "Ctor6" },
      f2: { kind: "Ctor8" },
    }
  );
  assert.deepEqual(
    _uniq_11(
      {
        kind: "Ctor9",
        f0: { kind: "Ctor8" },
        f1: { kind: "Ctor6" },
        f2: { kind: "Ctor8" },
      },
      { kind: "Ctor7", f0: { kind: "Ctor6" } }
    ),
    {
      kind: "Ctor9",
      f0: { kind: "Ctor8" },
      f1: { kind: "Ctor6" },
      f2: {
        kind: "Ctor9",
        f0: { kind: "Ctor8" },
        f1: { kind: "Ctor7", f0: { kind: "Ctor6" } },
        f2: { kind: "Ctor8" },
      },
    }
  );
  assert.deepEqual(
    _uniq_11(
      {
        kind: "Ctor9",
        f0: { kind: "Ctor8" },
        f1: { kind: "Ctor6" },
        f2: { kind: "Ctor8" },
      },
      { kind: "Ctor7", f0: { kind: "Ctor7", f0: { kind: "Ctor6" } } }
    ),
    {
      kind: "Ctor9",
      f0: { kind: "Ctor8" },
      f1: { kind: "Ctor6" },
      f2: {
        kind: "Ctor9",
        f0: { kind: "Ctor8" },
        f1: { kind: "Ctor7", f0: { kind: "Ctor7", f0: { kind: "Ctor6" } } },
        f2: { kind: "Ctor8" },
      },
    }
  );
  assert.deepEqual(
    _uniq_11(
      {
        kind: "Ctor9",
        f0: { kind: "Ctor8" },
        f1: { kind: "Ctor7", f0: { kind: "Ctor6" } },
        f2: { kind: "Ctor8" },
      },
      { kind: "Ctor6" }
    ),
    {
      kind: "Ctor9",
      f0: {
        kind: "Ctor9",
        f0: { kind: "Ctor8" },
        f1: { kind: "Ctor6" },
        f2: { kind: "Ctor8" },
      },
      f1: { kind: "Ctor7", f0: { kind: "Ctor6" } },
      f2: { kind: "Ctor8" },
    }
  );
  assert.deepEqual(
    _uniq_11(
      {
        kind: "Ctor9",
        f0: { kind: "Ctor8" },
        f1: { kind: "Ctor7", f0: { kind: "Ctor6" } },
        f2: { kind: "Ctor8" },
      },
      { kind: "Ctor7", f0: { kind: "Ctor6" } }
    ),
    {
      kind: "Ctor9",
      f0: { kind: "Ctor8" },
      f1: { kind: "Ctor7", f0: { kind: "Ctor6" } },
      f2: { kind: "Ctor8" },
    }
  );
  assert.deepEqual(
    _uniq_11(
      {
        kind: "Ctor9",
        f0: { kind: "Ctor8" },
        f1: { kind: "Ctor7", f0: { kind: "Ctor6" } },
        f2: { kind: "Ctor8" },
      },
      { kind: "Ctor7", f0: { kind: "Ctor7", f0: { kind: "Ctor6" } } }
    ),
    {
      kind: "Ctor9",
      f0: { kind: "Ctor8" },
      f1: { kind: "Ctor7", f0: { kind: "Ctor6" } },
      f2: {
        kind: "Ctor9",
        f0: { kind: "Ctor8" },
        f1: { kind: "Ctor7", f0: { kind: "Ctor7", f0: { kind: "Ctor6" } } },
        f2: { kind: "Ctor8" },
      },
    }
  );
  assert.deepEqual(
    _uniq_11(
      {
        kind: "Ctor9",
        f0: { kind: "Ctor8" },
        f1: { kind: "Ctor7", f0: { kind: "Ctor7", f0: { kind: "Ctor6" } } },
        f2: { kind: "Ctor8" },
      },
      { kind: "Ctor6" }
    ),
    {
      kind: "Ctor9",
      f0: {
        kind: "Ctor9",
        f0: { kind: "Ctor8" },
        f1: { kind: "Ctor6" },
        f2: { kind: "Ctor8" },
      },
      f1: { kind: "Ctor7", f0: { kind: "Ctor7", f0: { kind: "Ctor6" } } },
      f2: { kind: "Ctor8" },
    }
  );
  assert.deepEqual(
    _uniq_11(
      {
        kind: "Ctor9",
        f0: { kind: "Ctor8" },
        f1: { kind: "Ctor7", f0: { kind: "Ctor7", f0: { kind: "Ctor6" } } },
        f2: { kind: "Ctor8" },
      },
      { kind: "Ctor7", f0: { kind: "Ctor6" } }
    ),
    {
      kind: "Ctor9",
      f0: {
        kind: "Ctor9",
        f0: { kind: "Ctor8" },
        f1: { kind: "Ctor7", f0: { kind: "Ctor6" } },
        f2: { kind: "Ctor8" },
      },
      f1: { kind: "Ctor7", f0: { kind: "Ctor7", f0: { kind: "Ctor6" } } },
      f2: { kind: "Ctor8" },
    }
  );
  assert.deepEqual(
    _uniq_11(
      {
        kind: "Ctor9",
        f0: { kind: "Ctor8" },
        f1: { kind: "Ctor7", f0: { kind: "Ctor7", f0: { kind: "Ctor6" } } },
        f2: { kind: "Ctor8" },
      },
      { kind: "Ctor7", f0: { kind: "Ctor7", f0: { kind: "Ctor6" } } }
    ),
    {
      kind: "Ctor9",
      f0: { kind: "Ctor8" },
      f1: { kind: "Ctor7", f0: { kind: "Ctor7", f0: { kind: "Ctor6" } } },
      f2: { kind: "Ctor8" },
    }
  );
  assert.deepEqual(
    _uniq_11(
      {
        kind: "Ctor9",
        f0: {
          kind: "Ctor9",
          f0: { kind: "Ctor8" },
          f1: { kind: "Ctor6" },
          f2: { kind: "Ctor8" },
        },
        f1: { kind: "Ctor7", f0: { kind: "Ctor6" } },
        f2: { kind: "Ctor8" },
      },
      { kind: "Ctor6" }
    ),
    {
      kind: "Ctor9",
      f0: {
        kind: "Ctor9",
        f0: { kind: "Ctor8" },
        f1: { kind: "Ctor6" },
        f2: { kind: "Ctor8" },
      },
      f1: { kind: "Ctor7", f0: { kind: "Ctor6" } },
      f2: { kind: "Ctor8" },
    }
  );
  assert.deepEqual(
    _uniq_11(
      {
        kind: "Ctor9",
        f0: {
          kind: "Ctor9",
          f0: { kind: "Ctor8" },
          f1: { kind: "Ctor6" },
          f2: { kind: "Ctor8" },
        },
        f1: { kind: "Ctor7", f0: { kind: "Ctor6" } },
        f2: { kind: "Ctor8" },
      },
      { kind: "Ctor7", f0: { kind: "Ctor6" } }
    ),
    {
      kind: "Ctor9",
      f0: {
        kind: "Ctor9",
        f0: { kind: "Ctor8" },
        f1: { kind: "Ctor6" },
        f2: { kind: "Ctor8" },
      },
      f1: { kind: "Ctor7", f0: { kind: "Ctor6" } },
      f2: { kind: "Ctor8" },
    }
  );
  assert.deepEqual(
    _uniq_11(
      {
        kind: "Ctor9",
        f0: {
          kind: "Ctor9",
          f0: { kind: "Ctor8" },
          f1: { kind: "Ctor6" },
          f2: { kind: "Ctor8" },
        },
        f1: { kind: "Ctor7", f0: { kind: "Ctor6" } },
        f2: { kind: "Ctor8" },
      },
      { kind: "Ctor7", f0: { kind: "Ctor7", f0: { kind: "Ctor6" } } }
    ),
    {
      kind: "Ctor9",
      f0: {
        kind: "Ctor9",
        f0: { kind: "Ctor8" },
        f1: { kind: "Ctor6" },
        f2: { kind: "Ctor8" },
      },
      f1: { kind: "Ctor7", f0: { kind: "Ctor6" } },
      f2: {
        kind: "Ctor9",
        f0: { kind: "Ctor8" },
        f1: { kind: "Ctor7", f0: { kind: "Ctor7", f0: { kind: "Ctor6" } } },
        f2: { kind: "Ctor8" },
      },
    }
  );
  assert.deepEqual(
    _uniq_11(
      {
        kind: "Ctor9",
        f0: { kind: "Ctor8" },
        f1: { kind: "Ctor7", f0: { kind: "Ctor6" } },
        f2: {
          kind: "Ctor9",
          f0: { kind: "Ctor8" },
          f1: { kind: "Ctor7", f0: { kind: "Ctor7", f0: { kind: "Ctor6" } } },
          f2: { kind: "Ctor8" },
        },
      },
      { kind: "Ctor6" }
    ),
    {
      kind: "Ctor9",
      f0: {
        kind: "Ctor9",
        f0: { kind: "Ctor8" },
        f1: { kind: "Ctor6" },
        f2: { kind: "Ctor8" },
      },
      f1: { kind: "Ctor7", f0: { kind: "Ctor6" } },
      f2: {
        kind: "Ctor9",
        f0: { kind: "Ctor8" },
        f1: { kind: "Ctor7", f0: { kind: "Ctor7", f0: { kind: "Ctor6" } } },
        f2: { kind: "Ctor8" },
      },
    }
  );
  assert.deepEqual(
    _uniq_11(
      {
        kind: "Ctor9",
        f0: { kind: "Ctor8" },
        f1: { kind: "Ctor7", f0: { kind: "Ctor6" } },
        f2: {
          kind: "Ctor9",
          f0: { kind: "Ctor8" },
          f1: { kind: "Ctor7", f0: { kind: "Ctor7", f0: { kind: "Ctor6" } } },
          f2: { kind: "Ctor8" },
        },
      },
      { kind: "Ctor7", f0: { kind: "Ctor6" } }
    ),
    {
      kind: "Ctor9",
      f0: { kind: "Ctor8" },
      f1: { kind: "Ctor7", f0: { kind: "Ctor6" } },
      f2: {
        kind: "Ctor9",
        f0: { kind: "Ctor8" },
        f1: { kind: "Ctor7", f0: { kind: "Ctor7", f0: { kind: "Ctor6" } } },
        f2: { kind: "Ctor8" },
      },
    }
  );
}
assertions();

function validations() {
  assert.deepEqual(
    _uniq_11(
      {
        kind: "Ctor9",
        f0: { kind: "Ctor8" },
        f1: { kind: "Ctor6" },
        f2: {
          kind: "Ctor9",
          f0: { kind: "Ctor8" },
          f1: { kind: "Ctor7", f0: { kind: "Ctor6" } },
          f2: {
            kind: "Ctor9",
            f0: { kind: "Ctor8" },
            f1: { kind: "Ctor7", f0: { kind: "Ctor7", f0: { kind: "Ctor6" } } },
            f2: { kind: "Ctor8" },
          },
        },
      },
      {
        kind: "Ctor7",
        f0: { kind: "Ctor7", f0: { kind: "Ctor7", f0: { kind: "Ctor6" } } },
      }
    ),
    {
      kind: "Ctor9",
      f0: { kind: "Ctor8" },
      f1: { kind: "Ctor6" },
      f2: {
        kind: "Ctor9",
        f0: { kind: "Ctor8" },
        f1: { kind: "Ctor7", f0: { kind: "Ctor6" } },
        f2: {
          kind: "Ctor9",
          f0: { kind: "Ctor8" },
          f1: { kind: "Ctor7", f0: { kind: "Ctor7", f0: { kind: "Ctor6" } } },
          f2: {
            kind: "Ctor9",
            f0: { kind: "Ctor8" },
            f1: {
              kind: "Ctor7",
              f0: {
                kind: "Ctor7",
                f0: { kind: "Ctor7", f0: { kind: "Ctor6" } },
              },
            },
            f2: { kind: "Ctor8" },
          },
        },
      },
    }
  );
  assert.deepEqual(
    _uniq_11(
      {
        kind: "Ctor9",
        f0: {
          kind: "Ctor9",
          f0: {
            kind: "Ctor9",
            f0: { kind: "Ctor8" },
            f1: { kind: "Ctor7", f0: { kind: "Ctor6" } },
            f2: { kind: "Ctor8" },
          },
          f1: { kind: "Ctor7", f0: { kind: "Ctor7", f0: { kind: "Ctor6" } } },
          f2: { kind: "Ctor8" },
        },
        f1: {
          kind: "Ctor7",
          f0: { kind: "Ctor7", f0: { kind: "Ctor7", f0: { kind: "Ctor6" } } },
        },
        f2: { kind: "Ctor8" },
      },
      { kind: "Ctor6" }
    ),
    {
      kind: "Ctor9",
      f0: {
        kind: "Ctor9",
        f0: {
          kind: "Ctor9",
          f0: {
            kind: "Ctor9",
            f0: { kind: "Ctor8" },
            f1: { kind: "Ctor6" },
            f2: { kind: "Ctor8" },
          },
          f1: { kind: "Ctor7", f0: { kind: "Ctor6" } },
          f2: { kind: "Ctor8" },
        },
        f1: { kind: "Ctor7", f0: { kind: "Ctor7", f0: { kind: "Ctor6" } } },
        f2: { kind: "Ctor8" },
      },
      f1: {
        kind: "Ctor7",
        f0: { kind: "Ctor7", f0: { kind: "Ctor7", f0: { kind: "Ctor6" } } },
      },
      f2: { kind: "Ctor8" },
    }
  );
  assert.deepEqual(
    _uniq_11(
      {
        kind: "Ctor9",
        f0: {
          kind: "Ctor9",
          f0: { kind: "Ctor8" },
          f1: { kind: "Ctor7", f0: { kind: "Ctor7", f0: { kind: "Ctor6" } } },
          f2: { kind: "Ctor8" },
        },
        f1: {
          kind: "Ctor7",
          f0: {
            kind: "Ctor7",
            f0: { kind: "Ctor7", f0: { kind: "Ctor7", f0: { kind: "Ctor6" } } },
          },
        },
        f2: {
          kind: "Ctor9",
          f0: { kind: "Ctor8" },
          f1: {
            kind: "Ctor7",
            f0: {
              kind: "Ctor7",
              f0: {
                kind: "Ctor7",
                f0: {
                  kind: "Ctor7",
                  f0: { kind: "Ctor7", f0: { kind: "Ctor6" } },
                },
              },
            },
          },
          f2: { kind: "Ctor8" },
        },
      },
      {
        kind: "Ctor7",
        f0: { kind: "Ctor7", f0: { kind: "Ctor7", f0: { kind: "Ctor6" } } },
      }
    ),
    {
      kind: "Ctor9",
      f0: {
        kind: "Ctor9",
        f0: { kind: "Ctor8" },
        f1: { kind: "Ctor7", f0: { kind: "Ctor7", f0: { kind: "Ctor6" } } },
        f2: {
          kind: "Ctor9",
          f0: { kind: "Ctor8" },
          f1: {
            kind: "Ctor7",
            f0: { kind: "Ctor7", f0: { kind: "Ctor7", f0: { kind: "Ctor6" } } },
          },
          f2: { kind: "Ctor8" },
        },
      },
      f1: {
        kind: "Ctor7",
        f0: {
          kind: "Ctor7",
          f0: { kind: "Ctor7", f0: { kind: "Ctor7", f0: { kind: "Ctor6" } } },
        },
      },
      f2: {
        kind: "Ctor9",
        f0: { kind: "Ctor8" },
        f1: {
          kind: "Ctor7",
          f0: {
            kind: "Ctor7",
            f0: {
              kind: "Ctor7",
              f0: {
                kind: "Ctor7",
                f0: { kind: "Ctor7", f0: { kind: "Ctor6" } },
              },
            },
          },
        },
        f2: { kind: "Ctor8" },
      },
    }
  );
  assert.deepEqual(
    _uniq_11(
      {
        kind: "Ctor9",
        f0: { kind: "Ctor8" },
        f1: { kind: "Ctor7", f0: { kind: "Ctor7", f0: { kind: "Ctor6" } } },
        f2: {
          kind: "Ctor9",
          f0: { kind: "Ctor8" },
          f1: {
            kind: "Ctor7",
            f0: {
              kind: "Ctor7",
              f0: {
                kind: "Ctor7",
                f0: { kind: "Ctor7", f0: { kind: "Ctor6" } },
              },
            },
          },
          f2: { kind: "Ctor8" },
        },
      },
      {
        kind: "Ctor7",
        f0: { kind: "Ctor7", f0: { kind: "Ctor7", f0: { kind: "Ctor6" } } },
      }
    ),
    {
      kind: "Ctor9",
      f0: { kind: "Ctor8" },
      f1: { kind: "Ctor7", f0: { kind: "Ctor7", f0: { kind: "Ctor6" } } },
      f2: {
        kind: "Ctor9",
        f0: {
          kind: "Ctor9",
          f0: { kind: "Ctor8" },
          f1: {
            kind: "Ctor7",
            f0: { kind: "Ctor7", f0: { kind: "Ctor7", f0: { kind: "Ctor6" } } },
          },
          f2: { kind: "Ctor8" },
        },
        f1: {
          kind: "Ctor7",
          f0: {
            kind: "Ctor7",
            f0: { kind: "Ctor7", f0: { kind: "Ctor7", f0: { kind: "Ctor6" } } },
          },
        },
        f2: { kind: "Ctor8" },
      },
    }
  );
}
validations();
