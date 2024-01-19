declare var require: any;
const assert = require("node:assert");
type bbool = { kind: "True" } | { kind: "False" };

function bool_band(a: bbool): (b: bbool) => bbool {
  return function (b: bbool) {
    switch (b.kind) {
      case "True": {
        return a;
      }
      case "False": {
        return { kind: "False" };
      }
    }
  };
}

function assertions() {
  assert.deepEqual(bool_band({ kind: "True" })({ kind: "True" }), {
    kind: "True",
  });
  assert.deepEqual(bool_band({ kind: "True" })({ kind: "False" }), {
    kind: "False",
  });
  assert.deepEqual(bool_band({ kind: "False" })({ kind: "True" }), {
    kind: "False",
  });
  assert.deepEqual(bool_band({ kind: "False" })({ kind: "False" }), {
    kind: "False",
  });
}
assertions();

function validations() {
  assert.deepEqual(bool_band({ kind: "True" })({ kind: "True" }), {
    kind: "True",
  });
  assert.deepEqual(bool_band({ kind: "True" })({ kind: "False" }), {
    kind: "False",
  });
  assert.deepEqual(bool_band({ kind: "False" })({ kind: "True" }), {
    kind: "False",
  });
  assert.deepEqual(bool_band({ kind: "False" })({ kind: "False" }), {
    kind: "False",
  });
}
validations();
