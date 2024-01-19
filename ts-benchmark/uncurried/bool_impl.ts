declare var require: any;
const assert = require("node:assert");
type bbool = { kind: "True" } | { kind: "False" };

function bool_impl(x1: bbool, x2: bbool): bbool {
  switch (x1.kind) {
    case "True": {
      return x2;
    }
    case "False": {
      return { kind: "True" };
    }
  }
}

function assertions() {
  assert.deepEqual(bool_impl({ kind: "True" }, { kind: "True" }), {
    kind: "True",
  });
  assert.deepEqual(bool_impl({ kind: "True" }, { kind: "False" }), {
    kind: "False",
  });
  assert.deepEqual(bool_impl({ kind: "False" }, { kind: "True" }), {
    kind: "True",
  });
  assert.deepEqual(bool_impl({ kind: "False" }, { kind: "False" }), {
    kind: "True",
  });
}
assertions();

function validations() {
  assert.deepEqual(bool_impl({ kind: "True" }, { kind: "True" }), {
    kind: "True",
  });
  assert.deepEqual(bool_impl({ kind: "True" }, { kind: "False" }), {
    kind: "False",
  });
  assert.deepEqual(bool_impl({ kind: "False" }, { kind: "True" }), {
    kind: "True",
  });
  assert.deepEqual(bool_impl({ kind: "False" }, { kind: "False" }), {
    kind: "True",
  });
}
validations();
