declare var require: any;
const assert = require("node:assert");
type _uniq_0 = { kind: "Ctor4" } | { kind: "Ctor5"; f0: _uniq_0 };
type _uniq_1 =
  | { kind: "Ctor6" }
  | { kind: "Ctor7"; f0: _uniq_0; f1: _uniq_1; f2: _uniq_1 };
type _uniq_2 = { kind: "Ctor8" } | { kind: "Ctor9" };
type _uniq_3 = { kind: "Ctor10" } | { kind: "Ctor11" } | { kind: "Ctor12" };

function _uniq_13(_uniq_18: _uniq_0, _uniq_19: _uniq_0): _uniq_3 {
  switch (_uniq_18.kind) {
    case "Ctor4": {
      switch (_uniq_19.kind) {
        case "Ctor4": {
          return { kind: "Ctor11" };
        }
        case "Ctor5": {
          let _uniq_20 = _uniq_19.f0;
          return { kind: "Ctor10" };
        }
      }
    }
    case "Ctor5": {
      let _uniq_21 = _uniq_18.f0;
      switch (_uniq_19.kind) {
        case "Ctor4": {
          return { kind: "Ctor12" };
        }
        case "Ctor5": {
          let _uniq_22 = _uniq_19.f0;
          return _uniq_13(_uniq_21, _uniq_22);
        }
      }
    }
  }
}
function _uniq_14(_uniq_23: _uniq_0, _uniq_24: _uniq_0): _uniq_0 {
  switch (_uniq_13(_uniq_23, _uniq_24).kind) {
    case "Ctor10": {
      return _uniq_24;
    }
    case "Ctor11": {
      return _uniq_23;
    }
    case "Ctor12": {
      return _uniq_23;
    }
  }
}
function _uniq_15(_uniq_25: _uniq_1): _uniq_0 {
  switch (_uniq_25.kind) {
    case "Ctor6": {
      return { kind: "Ctor4" };
    }
    case "Ctor7": {
      let _uniq_28 = _uniq_25.f2;
      let _uniq_27 = _uniq_25.f1;
      let _uniq_26 = _uniq_25.f0;
      switch (_uniq_27.kind) {
        case "Ctor6": {
          return { kind: "Ctor5", f0: _uniq_15(_uniq_28) };
        }
        case "Ctor7": {
          let _uniq_31 = _uniq_27.f2;
          let _uniq_30 = _uniq_27.f1;
          let _uniq_29 = _uniq_27.f0;
          switch (_uniq_28.kind) {
            case "Ctor6": {
              return { kind: "Ctor5", f0: _uniq_15(_uniq_27) };
            }
            case "Ctor7": {
              let _uniq_34 = _uniq_28.f2;
              let _uniq_33 = _uniq_28.f1;
              let _uniq_32 = _uniq_28.f0;
              return {
                kind: "Ctor5",
                f0: _uniq_14(_uniq_15(_uniq_27), _uniq_15(_uniq_28)),
              };
            }
          }
        }
      }
    }
  }
}
function _uniq_16(_uniq_35: _uniq_2, _uniq_36: _uniq_2): _uniq_2 {
  switch (_uniq_35.kind) {
    case "Ctor8": {
      return _uniq_36;
    }
    case "Ctor9": {
      return { kind: "Ctor9" };
    }
  }
}

function _uniq_17(_uniq_37: _uniq_1): _uniq_2 {
  switch (_uniq_37.kind) {
    case "Ctor6": {
      return { kind: "Ctor8" };
    }
    case "Ctor7": {
      let _uniq_40 = _uniq_37.f2;
      let _uniq_39 = _uniq_37.f1;
      let _uniq_38 = _uniq_37.f0;
      let _uniq_41: _uniq_0 = _uniq_15(_uniq_39);
      let _uniq_42: _uniq_0 = _uniq_15(_uniq_40);
      switch (_uniq_13(_uniq_41, _uniq_42).kind) {
        case "Ctor11": {
          return _uniq_16(_uniq_17(_uniq_39), _uniq_17(_uniq_40));
        }
        case "Ctor10": {
          switch (_uniq_42.kind) {
            case "Ctor4": {
              return { kind: "Ctor9" };
            }
            case "Ctor5": {
              let _uniq_43 = _uniq_42.f0;
              switch (_uniq_13(_uniq_41, _uniq_43).kind) {
                case "Ctor11": {
                  return _uniq_16(_uniq_17(_uniq_39), _uniq_17(_uniq_40));
                }
                case "Ctor10": {
                  return { kind: "Ctor9" };
                }
                case "Ctor12": {
                  return { kind: "Ctor9" };
                }
              }
            }
          }
        }
        case "Ctor12": {
          switch (_uniq_41.kind) {
            case "Ctor4": {
              return { kind: "Ctor9" };
            }
            case "Ctor5": {
              let _uniq_44 = _uniq_41.f0;
              switch (_uniq_13(_uniq_42, _uniq_44).kind) {
                case "Ctor11": {
                  return _uniq_16(_uniq_17(_uniq_39), _uniq_17(_uniq_40));
                }
                case "Ctor10": {
                  return { kind: "Ctor9" };
                }
                case "Ctor12": {
                  return { kind: "Ctor9" };
                }
              }
            }
          }
        }
      }
    }
  }
}

function assertions() {
  assert.deepEqual(_uniq_17({ kind: "Ctor6" }), { kind: "Ctor8" });
  assert.deepEqual(
    _uniq_17({
      kind: "Ctor7",
      f0: { kind: "Ctor4" },
      f1: { kind: "Ctor6" },
      f2: { kind: "Ctor6" },
    }),
    { kind: "Ctor8" }
  );
  assert.deepEqual(
    _uniq_17({
      kind: "Ctor7",
      f0: { kind: "Ctor4" },
      f1: {
        kind: "Ctor7",
        f0: { kind: "Ctor5", f0: { kind: "Ctor4" } },
        f1: { kind: "Ctor6" },
        f2: { kind: "Ctor6" },
      },
      f2: { kind: "Ctor6" },
    }),
    { kind: "Ctor8" }
  );
  assert.deepEqual(
    _uniq_17({
      kind: "Ctor7",
      f0: { kind: "Ctor4" },
      f1: {
        kind: "Ctor7",
        f0: { kind: "Ctor5", f0: { kind: "Ctor4" } },
        f1: {
          kind: "Ctor7",
          f0: { kind: "Ctor5", f0: { kind: "Ctor4" } },
          f1: { kind: "Ctor6" },
          f2: { kind: "Ctor6" },
        },
        f2: { kind: "Ctor6" },
      },
      f2: { kind: "Ctor6" },
    }),
    { kind: "Ctor9" }
  );
  assert.deepEqual(
    _uniq_17({
      kind: "Ctor7",
      f0: { kind: "Ctor4" },
      f1: {
        kind: "Ctor7",
        f0: { kind: "Ctor5", f0: { kind: "Ctor4" } },
        f1: {
          kind: "Ctor7",
          f0: { kind: "Ctor5", f0: { kind: "Ctor4" } },
          f1: { kind: "Ctor6" },
          f2: { kind: "Ctor6" },
        },
        f2: {
          kind: "Ctor7",
          f0: { kind: "Ctor5", f0: { kind: "Ctor4" } },
          f1: { kind: "Ctor6" },
          f2: { kind: "Ctor6" },
        },
      },
      f2: {
        kind: "Ctor7",
        f0: { kind: "Ctor5", f0: { kind: "Ctor4" } },
        f1: { kind: "Ctor6" },
        f2: { kind: "Ctor6" },
      },
    }),
    { kind: "Ctor8" }
  );
  assert.deepEqual(
    _uniq_17({
      kind: "Ctor7",
      f0: { kind: "Ctor4" },
      f1: {
        kind: "Ctor7",
        f0: { kind: "Ctor5", f0: { kind: "Ctor4" } },
        f1: {
          kind: "Ctor7",
          f0: { kind: "Ctor5", f0: { kind: "Ctor4" } },
          f1: { kind: "Ctor6" },
          f2: { kind: "Ctor6" },
        },
        f2: {
          kind: "Ctor7",
          f0: { kind: "Ctor5", f0: { kind: "Ctor4" } },
          f1: { kind: "Ctor6" },
          f2: {
            kind: "Ctor7",
            f0: { kind: "Ctor5", f0: { kind: "Ctor4" } },
            f1: { kind: "Ctor6" },
            f2: { kind: "Ctor6" },
          },
        },
      },
      f2: {
        kind: "Ctor7",
        f0: { kind: "Ctor5", f0: { kind: "Ctor4" } },
        f1: { kind: "Ctor6" },
        f2: { kind: "Ctor6" },
      },
    }),
    { kind: "Ctor9" }
  );
}
assertions();

function validations() {
  assert.deepEqual(
    _uniq_17({
      kind: "Ctor7",
      f0: { kind: "Ctor4" },
      f1: {
        kind: "Ctor7",
        f0: { kind: "Ctor5", f0: { kind: "Ctor4" } },
        f1: {
          kind: "Ctor7",
          f0: { kind: "Ctor5", f0: { kind: "Ctor5", f0: { kind: "Ctor4" } } },
          f1: { kind: "Ctor6" },
          f2: { kind: "Ctor6" },
        },
        f2: { kind: "Ctor6" },
      },
      f2: { kind: "Ctor6" },
    }),
    { kind: "Ctor9" }
  );
  assert.deepEqual(
    _uniq_17({
      kind: "Ctor7",
      f0: { kind: "Ctor4" },
      f1: { kind: "Ctor6" },
      f2: {
        kind: "Ctor7",
        f0: { kind: "Ctor5", f0: { kind: "Ctor4" } },
        f1: {
          kind: "Ctor7",
          f0: { kind: "Ctor5", f0: { kind: "Ctor5", f0: { kind: "Ctor4" } } },
          f1: { kind: "Ctor6" },
          f2: { kind: "Ctor6" },
        },
        f2: { kind: "Ctor6" },
      },
    }),
    { kind: "Ctor9" }
  );
  assert.deepEqual(
    _uniq_17({
      kind: "Ctor7",
      f0: { kind: "Ctor5", f0: { kind: "Ctor5", f0: { kind: "Ctor4" } } },
      f1: {
        kind: "Ctor7",
        f0: {
          kind: "Ctor5",
          f0: { kind: "Ctor5", f0: { kind: "Ctor5", f0: { kind: "Ctor4" } } },
        },
        f1: { kind: "Ctor6" },
        f2: { kind: "Ctor6" },
      },
      f2: {
        kind: "Ctor7",
        f0: { kind: "Ctor4" },
        f1: { kind: "Ctor6" },
        f2: { kind: "Ctor6" },
      },
    }),
    { kind: "Ctor8" }
  );
  assert.deepEqual(
    _uniq_17({
      kind: "Ctor7",
      f0: {
        kind: "Ctor5",
        f0: { kind: "Ctor5", f0: { kind: "Ctor5", f0: { kind: "Ctor4" } } },
      },
      f1: {
        kind: "Ctor7",
        f0: { kind: "Ctor4" },
        f1: {
          kind: "Ctor7",
          f0: { kind: "Ctor5", f0: { kind: "Ctor4" } },
          f1: { kind: "Ctor6" },
          f2: { kind: "Ctor6" },
        },
        f2: {
          kind: "Ctor7",
          f0: {
            kind: "Ctor5",
            f0: {
              kind: "Ctor5",
              f0: {
                kind: "Ctor5",
                f0: { kind: "Ctor5", f0: { kind: "Ctor4" } },
              },
            },
          },
          f1: { kind: "Ctor6" },
          f2: { kind: "Ctor6" },
        },
      },
      f2: {
        kind: "Ctor7",
        f0: { kind: "Ctor5", f0: { kind: "Ctor4" } },
        f1: { kind: "Ctor6" },
        f2: { kind: "Ctor6" },
      },
    }),
    { kind: "Ctor8" }
  );
}
validations();
