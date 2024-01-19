/home/franlucc/.npm_packages/_npx/1bf7c3c15bf47d04/node_modules/ts-node/src/index.ts:859
    return new TSError(diagnosticText, diagnosticCodes, diagnostics);
           ^
TSError: ⨯ Unable to compile TypeScript:
[96mstarcoderbase-7b/multitok/prog_10/var_11.ts[0m:[93m9[0m:[93m57[0m - [91merror[0m[90m TS2366: [0mFunction lacks ending return statement and return type does not include 'undefined'.

[7m9[0m function _uniq_9(_uniq_11: _uniq_2, _uniq_12: _uniq_1): _uniq_1 {
[7m [0m [91m                                                        ~~~~~~~[0m
[96mstarcoderbase-7b/multitok/prog_10/var_11.ts[0m:[93m11[0m:[93m10[0m - [91merror[0m[90m TS2678: [0mType '"Ctor5"' is not comparable to type '"Ctor7" | "Ctor8"'.

[7m11[0m     case "Ctor5": {
[7m  [0m [91m         ~~~~~~~[0m
[96mstarcoderbase-7b/multitok/prog_10/var_11.ts[0m:[93m14[0m:[93m10[0m - [91merror[0m[90m TS2678: [0mType '"Ctor6"' is not comparable to type '"Ctor7" | "Ctor8"'.

[7m14[0m     case "Ctor6": {
[7m  [0m [91m         ~~~~~~~[0m
[96mstarcoderbase-7b/multitok/prog_10/var_11.ts[0m:[93m15[0m:[93m31[0m - [91merror[0m[90m TS2339: [0mProperty 'f0' does not exist on type 'never'.

[7m15[0m       let _uniq_13 = _uniq_11.f0;
[7m  [0m [91m                              ~~[0m
[96mstarcoderbase-7b/multitok/prog_10/var_11.ts[0m:[93m37[0m:[93m13[0m - [91merror[0m[90m TS2345: [0mArgument of type '_uniq_1' is not assignable to parameter of type '_uniq_2'.
  Type '{ kind: "Ctor5"; }' is not assignable to type '_uniq_2'.
    Type '{ kind: "Ctor5"; }' is missing the following properties from type '{ kind: "Ctor8"; f0: _uniq_2; f1: _uniq_0; f2: _uniq_2; }': f0, f1, f2

[7m37[0m             _uniq_10(_uniq_16, _uniq_19),
[7m  [0m [91m            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~[0m

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
