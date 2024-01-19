declare var require: any;
const assert = require("node:assert");
type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4"; f0: _uniq_0 };
type _uniq_1 =
  | { kind: "Ctor5"; f0: _uniq_0 }
  | { kind: "Ctor6"; f0: _uniq_1; f1: _uniq_1 }
  | { kind: "Ctor7"; f0: _uniq_1; f1: _uniq_1 }
  | { kind: "Ctor8"; f0: _uniq_1; f1: _uniq_1 }
  | { kind: "Ctor9"; f0: _uniq_1; f1: _uniq_1 };
type _uniq_2 = { kind: "Ctor10" } | { kind: "Ctor11" } | { kind: "Ctor12" };

function _uniq_13(_uniq_19: _uniq_0, _uniq_20: _uniq_0): _uniq_2 {
  switch (_uniq_19.kind) {
    case "Ctor3": {
      switch (_uniq_20.kind) {
        case "Ctor3": {
          return { kind: "Ctor11" };
        }
        case "Ctor4": {
          let _uniq_21 = _uniq_20.f0;
          return { kind: "Ctor10" };
        }
      }
    }
    case "Ctor4": {
      let _uniq_22 = _uniq_19.f0;
      switch (_uniq_20.kind) {
        case "Ctor3": {
          return { kind: "Ctor12" };
        }
        case "Ctor4": {
          let _uniq_23 = _uniq_20.f0;
          return _uniq_13(_uniq_22, _uniq_23);
        }
      }
    }
  }
}
function _uniq_14(_uniq_24: _uniq_0, _uniq_25: _uniq_0): _uniq_0 {
  switch (_uniq_24.kind) {
    case "Ctor3": {
      return _uniq_25;
    }
    case "Ctor4": {
      let _uniq_26 = _uniq_24.f0;
      return { kind: "Ctor4", f0: _uniq_14(_uniq_26, _uniq_25) };
    }
  }
}
function _uniq_15(_uniq_27: _uniq_0, _uniq_28: _uniq_0): _uniq_0 {
  switch (_uniq_27.kind) {
    case "Ctor3": {
      return { kind: "Ctor3" };
    }
    case "Ctor4": {
      let _uniq_29 = _uniq_27.f0;
      switch (_uniq_28.kind) {
        case "Ctor3": {
          return _uniq_27;
        }
        case "Ctor4": {
          let _uniq_30 = _uniq_28.f0;
          return _uniq_15(_uniq_29, _uniq_30);
        }
      }
    }
  }
}
function _uniq_16(_uniq_31: _uniq_0, _uniq_32: _uniq_0): _uniq_0 {
  switch (_uniq_31.kind) {
    case "Ctor3": {
      return { kind: "Ctor3" };
    }
    case "Ctor4": {
      let _uniq_33 = _uniq_31.f0;
      return _uniq_14(_uniq_32, _uniq_16(_uniq_33, _uniq_32));
    }
  }
}
function _uniq_17(_uniq_34: _uniq_0, _uniq_35: _uniq_0): _uniq_0 {
  switch (_uniq_35.kind) {
    case "Ctor3": {
      return { kind: "Ctor3" };
    }
    case "Ctor4": {
      let _uniq_36 = _uniq_35.f0;
      switch (_uniq_34.kind) {
        case "Ctor3": {
          return { kind: "Ctor3" };
        }
        case "Ctor4": {
          let _uniq_37 = _uniq_34.f0;
          switch (_uniq_13(_uniq_34, _uniq_35).kind) {
            case "Ctor10": {
              return { kind: "Ctor3" };
            }
            case "Ctor11": {
              return { kind: "Ctor4", f0: { kind: "Ctor3" } };
            }
            case "Ctor12": {
              return {
                kind: "Ctor4",
                f0: _uniq_17(_uniq_15(_uniq_34, _uniq_35), _uniq_35),
              };
            }
          }
        }
      }
    }
  }
}

function _uniq_18(_uniq_38: _uniq_1): _uniq_0 {
  switch (_uniq_38.kind) {
    case "Ctor5": {
      let _uniq_39 = _uniq_38.f0;
      return _uniq_39;
    }
    case "Ctor6": {
      let _uniq_41 = _uniq_38.f1;
      let _uniq_40 = _uniq_38.f0;
      return _uniq_14(_uniq_18(_uniq_40), _uniq_18(_uniq_41));
    }
    case "Ctor8": {
      let _uniq_43 = _uniq_38.f1;
      let _uniq_42 = _uniq_38.f0;
      return _uniq_16(_uniq_18(_uniq_42), _uniq_18(_uniq_43));
    }
    case "Ctor7": {
      let _uniq_45 = _uniq_38.f1;
      let _uniq_44 = _uniq_38.f0;
      return _uniq_15(_uniq_18(_uniq_44), _uniq_18(_uniq_45));
    }
    case "Ctor9": {
      let _uniq_47 = _uniq_38.f1;
      let _uniq_46 = _uniq_38.f0;
      return _uniq_17(_uniq_18(_uniq_46), _uniq_18(_uniq_47));
    }
  }
}

function assertions() {
  assert.deepEqual(
    _uniq_18({ kind: "Ctor5", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } }),
    { kind: "Ctor4", f0: { kind: "Ctor3" } }
  );
  assert.deepEqual(
    _uniq_18({
      kind: "Ctor6",
      f0: {
        kind: "Ctor5",
        f0: {
          kind: "Ctor4",
          f0: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
        },
      },
      f1: {
        kind: "Ctor5",
        f0: {
          kind: "Ctor4",
          f0: {
            kind: "Ctor4",
            f0: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
          },
        },
      },
    }),
    {
      kind: "Ctor4",
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
    }
  );
  assert.deepEqual(
    _uniq_18({
      kind: "Ctor8",
      f0: {
        kind: "Ctor5",
        f0: {
          kind: "Ctor4",
          f0: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
        },
      },
      f1: {
        kind: "Ctor5",
        f0: {
          kind: "Ctor4",
          f0: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
        },
      },
    }),
    {
      kind: "Ctor4",
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
        },
      },
    }
  );
  assert.deepEqual(
    _uniq_18({
      kind: "Ctor8",
      f0: {
        kind: "Ctor5",
        f0: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
      },
      f1: {
        kind: "Ctor5",
        f0: {
          kind: "Ctor4",
          f0: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
        },
      },
    }),
    {
      kind: "Ctor4",
      f0: {
        kind: "Ctor4",
        f0: {
          kind: "Ctor4",
          f0: {
            kind: "Ctor4",
            f0: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
          },
        },
      },
    }
  );
  assert.deepEqual(
    _uniq_18({
      kind: "Ctor7",
      f0: {
        kind: "Ctor5",
        f0: {
          kind: "Ctor4",
          f0: {
            kind: "Ctor4",
            f0: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
          },
        },
      },
      f1: {
        kind: "Ctor5",
        f0: {
          kind: "Ctor4",
          f0: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
        },
      },
    }),
    { kind: "Ctor4", f0: { kind: "Ctor3" } }
  );
  assert.deepEqual(
    _uniq_18({
      kind: "Ctor7",
      f0: {
        kind: "Ctor5",
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
      f1: { kind: "Ctor5", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
    }),
    {
      kind: "Ctor4",
      f0: {
        kind: "Ctor4",
        f0: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
      },
    }
  );
  assert.deepEqual(
    _uniq_18({
      kind: "Ctor9",
      f0: {
        kind: "Ctor5",
        f0: {
          kind: "Ctor4",
          f0: {
            kind: "Ctor4",
            f0: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
          },
        },
      },
      f1: {
        kind: "Ctor5",
        f0: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
      },
    }),
    { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } }
  );
  assert.deepEqual(
    _uniq_18({
      kind: "Ctor9",
      f0: {
        kind: "Ctor5",
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
        kind: "Ctor5",
        f0: {
          kind: "Ctor4",
          f0: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
        },
      },
    }),
    { kind: "Ctor4", f0: { kind: "Ctor3" } }
  );
}
assertions();

function validations() {
  assert.deepEqual(
    _uniq_18({
      kind: "Ctor6",
      f0: {
        kind: "Ctor5",
        f0: {
          kind: "Ctor4",
          f0: {
            kind: "Ctor4",
            f0: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
          },
        },
      },
      f1: {
        kind: "Ctor5",
        f0: {
          kind: "Ctor4",
          f0: {
            kind: "Ctor4",
            f0: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
          },
        },
      },
    }),
    {
      kind: "Ctor4",
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
                f0: {
                  kind: "Ctor4",
                  f0: { kind: "Ctor4", f0: { kind: "Ctor3" } },
                },
              },
            },
          },
        },
      },
    }
  );
  assert.deepEqual(
    _uniq_18({
      kind: "Ctor8",
      f0: {
        kind: "Ctor5",
        f0: {
          kind: "Ctor4",
          f0: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
        },
      },
      f1: {
        kind: "Ctor5",
        f0: {
          kind: "Ctor4",
          f0: {
            kind: "Ctor4",
            f0: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
          },
        },
      },
    }),
    {
      kind: "Ctor4",
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
              },
            },
          },
        },
      },
    }
  );
  assert.deepEqual(
    _uniq_18({
      kind: "Ctor9",
      f0: {
        kind: "Ctor5",
        f0: {
          kind: "Ctor4",
          f0: {
            kind: "Ctor4",
            f0: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
          },
        },
      },
      f1: {
        kind: "Ctor5",
        f0: {
          kind: "Ctor4",
          f0: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
        },
      },
    }),
    { kind: "Ctor4", f0: { kind: "Ctor3" } }
  );
}
validations();
