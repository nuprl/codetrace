declare var require: any;
const assert = require("node:assert");
type _uniq_0 = { kind: "Ctor1" } | { kind: "Ctor2"; f0: _uniq_0 };

function _uniq_3(_uniq_4: _uniq_0, _uniq_5: _uniq_0): _uniq_0 {
  switch (_uniq_4.kind) {
    case "Ctor1": {
      return _uniq_5;
    }
    case "Ctor2": {
      let _uniq_6 = _uniq_4.f0;
      return { kind: "Ctor2", f0: _uniq_3(_uniq_6, _uniq_5) };
    }
  }
}

function assertions() {
  assert.deepEqual(_uniq_3({ kind: "Ctor1" }, { kind: "Ctor1" }), {
    kind: "Ctor1",
  });
  assert.deepEqual(
    _uniq_3({ kind: "Ctor1" }, { kind: "Ctor2", f0: { kind: "Ctor1" } }),
    { kind: "Ctor2", f0: { kind: "Ctor1" } }
  );
  assert.deepEqual(
    _uniq_3({ kind: "Ctor2", f0: { kind: "Ctor1" } }, { kind: "Ctor1" }),
    { kind: "Ctor2", f0: { kind: "Ctor1" } }
  );
  assert.deepEqual(
    _uniq_3(
      { kind: "Ctor1" },
      { kind: "Ctor2", f0: { kind: "Ctor2", f0: { kind: "Ctor1" } } }
    ),
    { kind: "Ctor2", f0: { kind: "Ctor2", f0: { kind: "Ctor1" } } }
  );
  assert.deepEqual(
    _uniq_3(
      { kind: "Ctor2", f0: { kind: "Ctor1" } },
      { kind: "Ctor2", f0: { kind: "Ctor1" } }
    ),
    { kind: "Ctor2", f0: { kind: "Ctor2", f0: { kind: "Ctor1" } } }
  );
  assert.deepEqual(
    _uniq_3(
      { kind: "Ctor2", f0: { kind: "Ctor2", f0: { kind: "Ctor1" } } },
      { kind: "Ctor1" }
    ),
    { kind: "Ctor2", f0: { kind: "Ctor2", f0: { kind: "Ctor1" } } }
  );
  assert.deepEqual(
    _uniq_3(
      { kind: "Ctor2", f0: { kind: "Ctor1" } },
      { kind: "Ctor2", f0: { kind: "Ctor2", f0: { kind: "Ctor1" } } }
    ),
    {
      kind: "Ctor2",
      f0: { kind: "Ctor2", f0: { kind: "Ctor2", f0: { kind: "Ctor1" } } },
    }
  );
  assert.deepEqual(
    _uniq_3(
      { kind: "Ctor2", f0: { kind: "Ctor2", f0: { kind: "Ctor1" } } },
      { kind: "Ctor2", f0: { kind: "Ctor1" } }
    ),
    {
      kind: "Ctor2",
      f0: { kind: "Ctor2", f0: { kind: "Ctor2", f0: { kind: "Ctor1" } } },
    }
  );
}
assertions();

function validations() {
  assert.deepEqual(
    _uniq_3(
      { kind: "Ctor2", f0: { kind: "Ctor1" } },
      {
        kind: "Ctor2",
        f0: { kind: "Ctor2", f0: { kind: "Ctor2", f0: { kind: "Ctor1" } } },
      }
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
    _uniq_3(
      { kind: "Ctor2", f0: { kind: "Ctor2", f0: { kind: "Ctor1" } } },
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
          f0: { kind: "Ctor2", f0: { kind: "Ctor2", f0: { kind: "Ctor1" } } },
        },
      },
    }
  );
  assert.deepEqual(
    _uniq_3(
      { kind: "Ctor1" },
      {
        kind: "Ctor2",
        f0: { kind: "Ctor2", f0: { kind: "Ctor2", f0: { kind: "Ctor1" } } },
      }
    ),
    {
      kind: "Ctor2",
      f0: { kind: "Ctor2", f0: { kind: "Ctor2", f0: { kind: "Ctor1" } } },
    }
  );
  assert.deepEqual(
    _uniq_3(
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
            f0: { kind: "Ctor2", f0: { kind: "Ctor2", f0: { kind: "Ctor1" } } },
          },
        },
      },
    }
  );
}
validations();
