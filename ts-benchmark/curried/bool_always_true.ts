declare var require: any;
const assert = require("node:assert");
type bbool = { kind: "True" } | { kind: "False" };

function bool_always_true(b: bbool): bbool {
  return { kind: "True" };
}

function assertions() {
  assert.deepEqual(bool_always_true({ kind: "True" }), { kind: "True" });
  assert.deepEqual(bool_always_true({ kind: "False" }), { kind: "True" });
}
assertions();

function validations() {
  assert.deepEqual(bool_always_true({ kind: "True" }), { kind: "True" });
  assert.deepEqual(bool_always_true({ kind: "False" }), { kind: "True" });
}
validations();
