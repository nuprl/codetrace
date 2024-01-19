declare var require: any;
const assert = require("node:assert");
type bbool = { kind: "True" } | { kind: "False" };

function bool_xor(a: bbool): (b: bbool) => bbool {
  return function (b: bbool) {
    switch (a.kind) {
      case "True": {
        switch (b.kind) {
          case "False": {
            return { kind: "True" };
          }
          case "True": {
            return { kind: "False" };
          }
        }
      }
      case "False": {
        return b;
      }
    }
  };
}

function assertions() {
  assert.deepEqual(bool_xor({ kind: "True" })({ kind: "True" }), {
    kind: "False",
  });
  assert.deepEqual(bool_xor({ kind: "True" })({ kind: "False" }), {
    kind: "True",
  });
  assert.deepEqual(bool_xor({ kind: "False" })({ kind: "True" }), {
    kind: "True",
  });
  assert.deepEqual(bool_xor({ kind: "False" })({ kind: "False" }), {
    kind: "False",
  });
}
assertions();

function validations() {
  assert.deepEqual(bool_xor({ kind: "True" })({ kind: "True" }), {
    kind: "False",
  });
  assert.deepEqual(bool_xor({ kind: "True" })({ kind: "False" }), {
    kind: "True",
  });
  assert.deepEqual(bool_xor({ kind: "False" })({ kind: "True" }), {
    kind: "True",
  });
  assert.deepEqual(bool_xor({ kind: "False" })({ kind: "False" }), {
    kind: "False",
  });
}
validations();
