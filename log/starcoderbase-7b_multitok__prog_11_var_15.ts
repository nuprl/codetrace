/home/franlucc/.npm_packages/_npx/1bf7c3c15bf47d04/node_modules/ts-node/src/index.ts:859
    return new TSError(diagnosticText, diagnosticCodes, diagnostics);
           ^
TSError: ⨯ Unable to compile TypeScript:
[96mstarcoderbase-7b/multitok/prog_11/var_15.ts[0m:[93m20[0m:[93m57[0m - [91merror[0m[90m TS2366: [0mFunction lacks ending return statement and return type does not include 'undefined'.

[7m20[0m function _uniq_8(_uniq_13: _uniq_1, _uniq_14: _uniq_0): _uniq_0 {
[7m  [0m [91m                                                        ~~~~~~~[0m
[96mstarcoderbase-7b/multitok/prog_11/var_15.ts[0m:[93m22[0m:[93m10[0m - [91merror[0m[90m TS2678: [0mType '"Ctor2"' is not comparable to type '"Ctor4" | "Ctor5" | "Ctor6"'.

[7m22[0m     case "Ctor2": {
[7m  [0m [91m         ~~~~~~~[0m
[96mstarcoderbase-7b/multitok/prog_11/var_15.ts[0m:[93m25[0m:[93m10[0m - [91merror[0m[90m TS2678: [0mType '"Ctor3"' is not comparable to type '"Ctor4" | "Ctor5" | "Ctor6"'.

[7m25[0m     case "Ctor3": {
[7m  [0m [91m         ~~~~~~~[0m
[96mstarcoderbase-7b/multitok/prog_11/var_15.ts[0m:[93m26[0m:[93m31[0m - [91merror[0m[90m TS2339: [0mProperty 'f0' does not exist on type 'never'.

[7m26[0m       let _uniq_15 = _uniq_13.f0;
[7m  [0m [91m                              ~~[0m
[96mstarcoderbase-7b/multitok/prog_11/var_15.ts[0m:[93m46[0m:[93m22[0m - [91merror[0m[90m TS2345: [0mArgument of type '_uniq_0' is not assignable to parameter of type '_uniq_1'.
  Type '{ kind: "Ctor2"; }' is not assignable to type '_uniq_1'.
    Type '{ kind: "Ctor2"; }' is missing the following properties from type '{ kind: "Ctor6"; f0: _uniq_1; f1: _uniq_1; }': f0, f1

[7m46[0m       return _uniq_8(_uniq_9(_uniq_20), _uniq_9(_uniq_21));
[7m  [0m [91m                     ~~~~~~~~~~~~~~~~~[0m

    at createTSError (/home/franlucc/.npm_packages/_npx/1bf7c3c15bf47d04/node_modules/ts-node/src/index.ts:859:12)
    at reportTSError (/home/franlucc/.npm_packages/_npx/1bf7c3c15bf47d04/node_modules/ts-node/src/index.ts:863:19)
    at getOutput (/home/franlucc/.npm_packages/_npx/1bf7c3c15bf47d04/node_modules/ts-node/src/index.ts:1077:36)
    at Object.compile (/home/franlucc/.npm_packages/_npx/1bf7c3c15bf47d04/node_modules/ts-node/src/index.ts:1433:41)
    at Module.m._compile (/home/franlucc/.npm_packages/_npx/1bf7c3c15bf47d04/node_modules/ts-node/src/index.ts:1617:30)
    at Module._extensions..js (node:internal/modules/cjs/loader:1435:10)
    at Object.require.extensions.<computed> [as .ts] (/home/franlucc/.npm_packages/_npx/1bf7c3c15bf47d04/node_modules/ts-node/src/index.ts:1621:12)
    at Module.load (node:internal/modules/cjs/loader:1207:32)
    at Function.Module._load (node:internal/modules/cjs/loader:1023:12)
    at Function.executeUserEntryPoint [as runMain] (node:internal/modules/run_main:135:12) {
  diagnosticCodes: [ 2366, 2678, 2678, 2339, 2345 ]
}
