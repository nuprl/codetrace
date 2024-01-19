declare var require: any;
const assert = require("node:assert");
type _uniq_0 = { kind: "Ctor2" } | { kind: "Ctor3"; f0: _uniq_0 };
type _uniq_1 =
  | { kind: "Ctor4"; f0: _uniq_0 }
  | { kind: "Ctor5"; f0: _uniq_1; f1: _uniq_1 }
  | { kind: "Ctor6"; f0: _uniq_1; f1: _uniq_1 };

function _uniq_7(_uniq_10: _uniq_0): (_uniq_11: _uniq_0) => _uniq_0 {
  return function (_uniq_11: _uniq_0) {
    switch (_uniq_10.kind) {
      case "Ctor2": {
        return _uniq_11;
      }
      case "Ctor3": {
        let _uniq_12 = _uniq_10.f0;
        return { kind: "Ctor3", f0: _uniq_7(_uniq_12)(_uniq_11) };
      }
    }
  };
}
function _uniq_8(_uniq_13: _uniq_0): (_uniq_14: _uniq_0) => _uniq_0 {
  return function (_uniq_14: _uniq_0) {
    switch (_uniq_13.kind) {
      case "Ctor2": {
        return { kind: "Ctor2" };
      }
      case "Ctor3": {
        let _uniq_15 = _uniq_13.f0;
        return _uniq_7(_uniq_14)(_uniq_8(_uniq_15)(_uniq_14));
      }
    }
  };
}

function _uniq_9(_uniq_16: _uniq_1): _uniq_0 {
  switch (_uniq_16.kind) {
    case "Ctor4": {
      let _uniq_17 = _uniq_16.f0;
      return _uniq_17;
    }
    case "Ctor5": {
      let _uniq_19 = _uniq_16.f1;
      let _uniq_18 = _uniq_16.f0;
      return _uniq_7(_uniq_9(_uniq_18))(_uniq_9(_uniq_19));
    }
    case "Ctor6": {
      let _uniq_21 = _uniq_16.f1;
      let _uniq_20 = _uniq_16.f0;
      return _uniq_8(_uniq_9(_uniq_20))(_uniq_9(_uniq_21));
    }
  }
}

function assertions() {
  assert.deepEqual(
    _uniq_9({ kind: "Ctor4", f0: { kind: "Ctor3", f0: { kind: "Ctor2" } } }),
    { kind: "Ctor3", f0: { kind: "Ctor2" } }
  );
  assert.deepEqual(
    _uniq_9({
      kind: "Ctor5",
      f0: {
        kind: "Ctor4",
        f0: {
          kind: "Ctor3",
          f0: { kind: "Ctor3", f0: { kind: "Ctor3", f0: { kind: "Ctor2" } } },
        },
      },
      f1: {
        kind: "Ctor4",
        f0: {
          kind: "Ctor3",
          f0: {
            kind: "Ctor3",
            f0: { kind: "Ctor3", f0: { kind: "Ctor3", f0: { kind: "Ctor2" } } },
          },
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
    _uniq_9({
      kind: "Ctor6",
      f0: {
        kind: "Ctor4",
        f0: {
          kind: "Ctor3",
          f0: { kind: "Ctor3", f0: { kind: "Ctor3", f0: { kind: "Ctor2" } } },
        },
      },
      f1: {
        kind: "Ctor4",
        f0: {
          kind: "Ctor3",
          f0: { kind: "Ctor3", f0: { kind: "Ctor3", f0: { kind: "Ctor2" } } },
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
        },
      },
    }
  );
  assert.deepEqual(
    _uniq_9({
      kind: "Ctor6",
      f0: {
        kind: "Ctor4",
        f0: { kind: "Ctor3", f0: { kind: "Ctor3", f0: { kind: "Ctor2" } } },
      },
      f1: {
        kind: "Ctor4",
        f0: {
          kind: "Ctor3",
          f0: { kind: "Ctor3", f0: { kind: "Ctor3", f0: { kind: "Ctor2" } } },
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
            f0: { kind: "Ctor3", f0: { kind: "Ctor3", f0: { kind: "Ctor2" } } },
          },
        },
      },
    }
  );
}
assertions();

function validations() {
  assert.deepEqual(
    _uniq_9({
      kind: "Ctor5",
      f0: {
        kind: "Ctor4",
        f0: {
          kind: "Ctor3",
          f0: {
            kind: "Ctor3",
            f0: { kind: "Ctor3", f0: { kind: "Ctor3", f0: { kind: "Ctor2" } } },
          },
        },
      },
      f1: {
        kind: "Ctor4",
        f0: {
          kind: "Ctor3",
          f0: {
            kind: "Ctor3",
            f0: { kind: "Ctor3", f0: { kind: "Ctor3", f0: { kind: "Ctor2" } } },
          },
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
                f0: {
                  kind: "Ctor3",
                  f0: { kind: "Ctor3", f0: { kind: "Ctor2" } },
                },
              },
            },
          },
        },
      },
    }
  );
  assert.deepEqual(
    _uniq_9({
      kind: "Ctor6",
      f0: {
        kind: "Ctor4",
        f0: {
          kind: "Ctor3",
          f0: { kind: "Ctor3", f0: { kind: "Ctor3", f0: { kind: "Ctor2" } } },
        },
      },
      f1: {
        kind: "Ctor4",
        f0: {
          kind: "Ctor3",
          f0: {
            kind: "Ctor3",
            f0: { kind: "Ctor3", f0: { kind: "Ctor3", f0: { kind: "Ctor2" } } },
          },
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
              },
            },
          },
        },
      },
    }
  );
}
validations();
