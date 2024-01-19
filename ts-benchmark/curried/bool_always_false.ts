declare var require: any;
const assert = require("node:assert");
type bbool = { kind: "True" } | { kind: "False" };

function bool_always_false(b: bbool): bbool {
  return { kind: "False" };
}

function assertions() {
  assert.deepEqual(bool_always_false({ kind: "False" }), { kind: "False" });
  assert.deepEqual(bool_always_false({ kind: "True" }), { kind: "False" });
}
assertions();

function validations() {
  assert.deepEqual(bool_always_false({ kind: "False" }), { kind: "False" });
  assert.deepEqual(bool_always_false({ kind: "True" }), { kind: "False" });
}
validations();
