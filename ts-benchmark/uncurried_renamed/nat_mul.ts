declare var require: any;
const assert = require("node:assert");
type _uniq_0 = { kind: "Ctor1" } | { kind: "Ctor2"; f0: _uniq_0 };

function _uniq_3(_uniq_5: _uniq_0, _uniq_6: _uniq_0): _uniq_0 {
  switch (_uniq_5.kind) {
    case "Ctor1": {
      return _uniq_6;
    }
    case "Ctor2": {
      let _uniq_7 = _uniq_5.f0;
      return { kind: "Ctor2", f0: _uniq_3(_uniq_7, _uniq_6) };
    }
  }
}

function _uniq_4(_uniq_8: _uniq_0, _uniq_9: _uniq_0): _uniq_0 {
  switch (_uniq_8.kind) {
    case "Ctor1": {
      return { kind: "Ctor1" };
    }
    case "Ctor2": {
      let _uniq_10 = _uniq_8.f0;
      return _uniq_3(_uniq_9, _uniq_4(_uniq_10, _uniq_9));
    }
  }
}

function assertions() {
  assert.deepEqual(_uniq_4({ kind: "Ctor1" }, { kind: "Ctor1" }), {
    kind: "Ctor1",
  });
  assert.deepEqual(
    _uniq_4({ kind: "Ctor2", f0: { kind: "Ctor1" } }, { kind: "Ctor1" }),
    { kind: "Ctor1" }
  );
  assert.deepEqual(
    _uniq_4(
      { kind: "Ctor2", f0: { kind: "Ctor1" } },
      { kind: "Ctor2", f0: { kind: "Ctor2", f0: { kind: "Ctor1" } } }
    ),
    { kind: "Ctor2", f0: { kind: "Ctor2", f0: { kind: "Ctor1" } } }
  );
  assert.deepEqual(
    _uniq_4(
      {
        kind: "Ctor2",
        f0: { kind: "Ctor2", f0: { kind: "Ctor2", f0: { kind: "Ctor1" } } },
      },
      {
        kind: "Ctor2",
        f0: { kind: "Ctor2", f0: { kind: "Ctor2", f0: { kind: "Ctor1" } } },
      }
    ),
    {
      kind: "Ctor2",
      f0: {
        kind: "Ctor2",
        f0: {
          kind: "Ctor2",
          f0: {
            kind: "Ctor2",
            f0: {
              kind: "Ctor2",
              f0: {
                kind: "Ctor2",
                f0: {
                  kind: "Ctor2",
                  f0: {
                    kind: "Ctor2",
                    f0: { kind: "Ctor2", f0: { kind: "Ctor1" } },
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
    _uniq_4(
      {
        kind: "Ctor2",
        f0: { kind: "Ctor2", f0: { kind: "Ctor2", f0: { kind: "Ctor1" } } },
      },
      { kind: "Ctor2", f0: { kind: "Ctor2", f0: { kind: "Ctor1" } } }
    ),
    {
      kind: "Ctor2",
      f0: {
        kind: "Ctor2",
        f0: {
          kind: "Ctor2",
          f0: {
            kind: "Ctor2",
            f0: { kind: "Ctor2", f0: { kind: "Ctor2", f0: { kind: "Ctor1" } } },
          },
        },
      },
    }
  );
  assert.deepEqual(
    _uniq_4(
      { kind: "Ctor2", f0: { kind: "Ctor2", f0: { kind: "Ctor1" } } },
      { kind: "Ctor2", f0: { kind: "Ctor2", f0: { kind: "Ctor1" } } }
    ),
    {
      kind: "Ctor2",
      f0: {
        kind: "Ctor2",
        f0: { kind: "Ctor2", f0: { kind: "Ctor2", f0: { kind: "Ctor1" } } },
      },
    }
  );
}
assertions();

function validations() {
  assert.deepEqual(
    _uniq_4(
      { kind: "Ctor2", f0: { kind: "Ctor2", f0: { kind: "Ctor1" } } },
      { kind: "Ctor2", f0: { kind: "Ctor2", f0: { kind: "Ctor1" } } }
    ),
    {
      kind: "Ctor2",
      f0: {
        kind: "Ctor2",
        f0: { kind: "Ctor2", f0: { kind: "Ctor2", f0: { kind: "Ctor1" } } },
      },
    }
  );
  assert.deepEqual(
    _uniq_4(
      { kind: "Ctor1" },
      { kind: "Ctor2", f0: { kind: "Ctor2", f0: { kind: "Ctor1" } } }
    ),
    { kind: "Ctor1" }
  );
  assert.deepEqual(
    _uniq_4(
      {
        kind: "Ctor2",
        f0: { kind: "Ctor2", f0: { kind: "Ctor2", f0: { kind: "Ctor1" } } },
      },
      {
        kind: "Ctor2",
        f0: {
          kind: "Ctor2",
          f0: { kind: "Ctor2", f0: { kind: "Ctor2", f0: { kind: "Ctor1" } } },
        },
      }
    ),
    {
      kind: "Ctor2",
      f0: {
        kind: "Ctor2",
        f0: {
          kind: "Ctor2",
          f0: {
            kind: "Ctor2",
            f0: {
              kind: "Ctor2",
              f0: {
                kind: "Ctor2",
                f0: {
                  kind: "Ctor2",
                  f0: {
                    kind: "Ctor2",
                    f0: {
                      kind: "Ctor2",
                      f0: {
                        kind: "Ctor2",
                        f0: {
                          kind: "Ctor2",
                          f0: { kind: "Ctor2", f0: { kind: "Ctor1" } },
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
  assert.deepEqual(
    _uniq_4(
      { kind: "Ctor2", f0: { kind: "Ctor1" } },
      {
        kind: "Ctor2",
        f0: {
          kind: "Ctor2",
          f0: {
            kind: "Ctor2",
            f0: { kind: "Ctor2", f0: { kind: "Ctor2", f0: { kind: "Ctor1" } } },
          },
        },
      }
    ),
    {
      kind: "Ctor2",
      f0: {
        kind: "Ctor2",
        f0: {
          kind: "Ctor2",
          f0: { kind: "Ctor2", f0: { kind: "Ctor2", f0: { kind: "Ctor1" } } },
        },
      },
    }
  );
}
validations();
