declare var require: any;
const assert = require("node:assert");
type bbool = { kind: "True" } | { kind: "False" };
type formula =
  | { kind: "Bool"; f0: bbool }
  | { kind: "Not"; f0: formula }
  | { kind: "AndAlso"; f0: formula; f1: formula }
  | { kind: "OrElse"; f0: formula; f1: formula }
  | { kind: "Imply"; f0: formula; f1: formula };

function bool_not(b: bbool): bbool {
  switch (b.kind) {
    case "True": {
      return { kind: "False" };
    }
    case "False": {
      return { kind: "True" };
    }
  }
}
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
function bool_bor(a: bbool): (b: bbool) => bbool {
  return function (b: bbool) {
    switch (a.kind) {
      case "True": {
        return { kind: "True" };
      }
      case "False": {
        return b;
      }
    }
  };
}

function expr_eval(f: formula): bbool {
  switch (f.kind) {
    case "Bool": {
      let b = f.f0;
      return b;
    }
    case "Not": {
      let f1 = f.f0;
      return bool_not(expr_eval(f1));
    }
    case "AndAlso": {
      let f2 = f.f1;
      let f1 = f.f0;
      return bool_band(expr_eval(f1))(expr_eval(f2));
    }
    case "OrElse": {
      let f2 = f.f1;
      let f1 = f.f0;
      return bool_bor(expr_eval(f1))(expr_eval(f2));
    }
    case "Imply": {
      let f2 = f.f1;
      let f1 = f.f0;
      return bool_bor(bool_not(expr_eval(f1)))(expr_eval(f2));
    }
  }
}

function assertions() {
  assert.deepEqual(expr_eval({ kind: "Bool", f0: { kind: "True" } }), {
    kind: "True",
  });
  assert.deepEqual(expr_eval({ kind: "Bool", f0: { kind: "False" } }), {
    kind: "False",
  });
  assert.deepEqual(
    expr_eval({ kind: "Not", f0: { kind: "Bool", f0: { kind: "True" } } }),
    { kind: "False" }
  );
  assert.deepEqual(
    expr_eval({ kind: "Not", f0: { kind: "Bool", f0: { kind: "False" } } }),
    { kind: "True" }
  );
  assert.deepEqual(
    expr_eval({
      kind: "AndAlso",
      f0: { kind: "Bool", f0: { kind: "True" } },
      f1: { kind: "Bool", f0: { kind: "False" } },
    }),
    { kind: "False" }
  );
  assert.deepEqual(
    expr_eval({
      kind: "AndAlso",
      f0: { kind: "Bool", f0: { kind: "True" } },
      f1: { kind: "Bool", f0: { kind: "True" } },
    }),
    { kind: "True" }
  );
  assert.deepEqual(
    expr_eval({
      kind: "AndAlso",
      f0: { kind: "Bool", f0: { kind: "False" } },
      f1: { kind: "Bool", f0: { kind: "True" } },
    }),
    { kind: "False" }
  );
  assert.deepEqual(
    expr_eval({
      kind: "OrElse",
      f0: { kind: "Bool", f0: { kind: "True" } },
      f1: { kind: "Bool", f0: { kind: "False" } },
    }),
    { kind: "True" }
  );
  assert.deepEqual(
    expr_eval({
      kind: "OrElse",
      f0: { kind: "Bool", f0: { kind: "False" } },
      f1: { kind: "Bool", f0: { kind: "False" } },
    }),
    { kind: "False" }
  );
  assert.deepEqual(
    expr_eval({
      kind: "OrElse",
      f0: { kind: "Bool", f0: { kind: "False" } },
      f1: { kind: "Bool", f0: { kind: "True" } },
    }),
    { kind: "True" }
  );
  assert.deepEqual(
    expr_eval({
      kind: "Imply",
      f0: { kind: "Bool", f0: { kind: "True" } },
      f1: { kind: "Bool", f0: { kind: "False" } },
    }),
    { kind: "False" }
  );
  assert.deepEqual(
    expr_eval({
      kind: "Imply",
      f0: { kind: "Bool", f0: { kind: "False" } },
      f1: { kind: "Bool", f0: { kind: "False" } },
    }),
    { kind: "True" }
  );
  assert.deepEqual(
    expr_eval({
      kind: "Imply",
      f0: { kind: "Bool", f0: { kind: "False" } },
      f1: { kind: "Bool", f0: { kind: "True" } },
    }),
    { kind: "True" }
  );
  assert.deepEqual(
    expr_eval({
      kind: "Imply",
      f0: { kind: "Bool", f0: { kind: "True" } },
      f1: { kind: "Bool", f0: { kind: "True" } },
    }),
    { kind: "True" }
  );
}
assertions();

function validations() {
  assert.deepEqual(
    expr_eval({
      kind: "Imply",
      f0: { kind: "Bool", f0: { kind: "True" } },
      f1: {
        kind: "AndAlso",
        f0: { kind: "Bool", f0: { kind: "True" } },
        f1: { kind: "Bool", f0: { kind: "False" } },
      },
    }),
    { kind: "False" }
  );
  assert.deepEqual(
    expr_eval({
      kind: "Imply",
      f0: {
        kind: "OrElse",
        f0: { kind: "Bool", f0: { kind: "True" } },
        f1: { kind: "Bool", f0: { kind: "False" } },
      },
      f1: { kind: "Bool", f0: { kind: "True" } },
    }),
    { kind: "True" }
  );
  assert.deepEqual(
    expr_eval({
      kind: "Imply",
      f0: {
        kind: "OrElse",
        f0: { kind: "Bool", f0: { kind: "True" } },
        f1: { kind: "Bool", f0: { kind: "False" } },
      },
      f1: {
        kind: "OrElse",
        f0: { kind: "Bool", f0: { kind: "True" } },
        f1: { kind: "Bool", f0: { kind: "False" } },
      },
    }),
    { kind: "True" }
  );
}
validations();
