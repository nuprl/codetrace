declare var require: any;
const assert = require("node:assert");
type bbool = { kind: "True" } | { kind: "False" };

function bool_bor(a: bbool, b: bbool): bbool {
  switch (a.kind) {
    case "True": {
      return { kind: "True" };
    }
    case "False": {
      return b;
    }
  }
}

function assertions() {
  assert.deepEqual(bool_bor({ kind: "True" }, { kind: "True" }), {
    kind: "True",
  });
  assert.deepEqual(bool_bor({ kind: "True" }, { kind: "False" }), {
    kind: "True",
  });
  assert.deepEqual(bool_bor({ kind: "False" }, { kind: "True" }), {
    kind: "True",
  });
  assert.deepEqual(bool_bor({ kind: "False" }, { kind: "False" }), {
    kind: "False",
  });
}
assertions();

function validations() {
  assert.deepEqual(bool_bor({ kind: "True" }, { kind: "True" }), {
    kind: "True",
  });
  assert.deepEqual(bool_bor({ kind: "True" }, { kind: "False" }), {
    kind: "True",
  });
  assert.deepEqual(bool_bor({ kind: "False" }, { kind: "True" }), {
    kind: "True",
  });
  assert.deepEqual(bool_bor({ kind: "False" }, { kind: "False" }), {
    kind: "False",
  });
}
validations();
