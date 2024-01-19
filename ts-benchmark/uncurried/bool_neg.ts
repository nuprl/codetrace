declare var require: any;
const assert = require("node:assert");
type bbool = { kind: "True" } | { kind: "False" };

function bool_neg(a: bbool): bbool {
  switch (a.kind) {
    case "True": {
      return { kind: "False" };
    }
    case "False": {
      return { kind: "True" };
    }
  }
}

function assertions() {
  assert.deepEqual(bool_neg({ kind: "True" }), { kind: "False" });
  assert.deepEqual(bool_neg({ kind: "False" }), { kind: "True" });
}
assertions();

function validations() {
  assert.deepEqual(bool_neg({ kind: "True" }), { kind: "False" });
  assert.deepEqual(bool_neg({ kind: "False" }), { kind: "True" });
}
validations();
