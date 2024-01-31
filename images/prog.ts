declare var require: any;
const assert = require("node:assert");
type _uniq_q = { kind: "Ctor2" } | { kind: "Ctor3"; f0: _uniq_q };
type _uniq_t= { kind: "Ctor4" } | { kind: "Ctor5"; f0: _uniq_q; f1: _uniq_t};

function _uniq_6(
  _uniq_9: _uniq_t
  _uniq_10: (__x7: _uniq_q, __x8: <FILL>) => _uniq_q,
  _uniq_11: _uniq_q
): _uniq_q {
  switch (_uniq_9.kind) {
    case "Ctor4": {
      return _uniq_11;
    }
    case "Ctor5": {
      let _uniq_13 = _uniq_9.f1;
      let _uniq_12 = _uniq_9.f0;
      return _uniq_6(_uniq_13, _uniq_10, _uniq_10(_uniq_11, _uniq_12));
    }
  }
}
// answer is _uniq_q, starcoderbase-1b returns uniq_t