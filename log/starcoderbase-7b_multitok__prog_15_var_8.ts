/home/franlucc/.npm_packages/_npx/1bf7c3c15bf47d04/node_modules/ts-node/src/index.ts:859
    return new TSError(diagnosticText, diagnosticCodes, diagnostics);
           ^
TSError: ⨯ Unable to compile TypeScript:
[96mstarcoderbase-7b/multitok/prog_15/var_8.ts[0m:[93m5[0m:[93m72[0m - [91merror[0m[90m TS2693: [0m'_uniq_2' only refers to a type, but is being used as a value here.

[7m5[0m type _uniq_2 = { kind: "Ctor7" } | { kind: "Ctor8"; f0: _uniq_1 }; f1: _uniq_2 };
[7m [0m [91m                                                                       ~~~~~~~[0m
[96mstarcoderbase-7b/multitok/prog_15/var_8.ts[0m:[93m13[0m:[93m31[0m - [91merror[0m[90m TS2339: [0mProperty 'f1' does not exist on type '{ kind: "Ctor8"; f0: _uniq_1; }'.

[7m13[0m       let _uniq_12 = _uniq_10.f1;
[7m  [0m [91m                              ~~[0m
[96mstarcoderbase-7b/multitok/prog_15/var_8.ts[0m:[93m38[0m:[93m53[0m - [91merror[0m[90m TS2353: [0mObject literal may only specify known properties, and 'f1' does not exist in type '{ kind: "Ctor8"; f0: _uniq_1; }'.

[7m38[0m     _uniq_9({ kind: "Ctor8", f0: { kind: "Ctor6" }, f1: { kind: "Ctor7" } }),
[7m  [0m [91m                                                    ~~[0m
[96mstarcoderbase-7b/multitok/prog_15/var_8.ts[0m:[93m42[0m:[93m53[0m - [91merror[0m[90m TS2353: [0mObject literal may only specify known properties, and 'f1' does not exist in type '{ kind: "Ctor8"; f0: _uniq_1; }'.

[7m42[0m     _uniq_9({ kind: "Ctor8", f0: { kind: "Ctor5" }, f1: { kind: "Ctor7" } }),
[7m  [0m [91m                                                    ~~[0m
[96mstarcoderbase-7b/multitok/prog_15/var_8.ts[0m:[93m49[0m:[93m7[0m - [91merror[0m[90m TS2353: [0mObject literal may only specify known properties, and 'f1' does not exist in type '{ kind: "Ctor8"; f0: _uniq_1; }'.

[7m49[0m       f1: { kind: "Ctor8", f0: { kind: "Ctor6" }, f1: { kind: "Ctor7" } },
[7m  [0m [91m      ~~[0m
[96mstarcoderbase-7b/multitok/prog_15/var_8.ts[0m:[93m57[0m:[93m7[0m - [91merror[0m[90m TS2353: [0mObject literal may only specify known properties, and 'f1' does not exist in type '{ kind: "Ctor8"; f0: _uniq_1; }'.

[7m57[0m       f1: { kind: "Ctor8", f0: { kind: "Ctor5" }, f1: { kind: "Ctor7" } },
[7m  [0m [91m      ~~[0m
[96mstarcoderbase-7b/multitok/prog_15/var_8.ts[0m:[93m65[0m:[93m7[0m - [91merror[0m[90m TS2353: [0mObject literal may only specify known properties, and 'f1' does not exist in type '{ kind: "Ctor8"; f0: _uniq_1; }'.

[7m65[0m       f1: { kind: "Ctor8", f0: { kind: "Ctor6" }, f1: { kind: "Ctor7" } },
[7m  [0m [91m      ~~[0m
[96mstarcoderbase-7b/multitok/prog_15/var_8.ts[0m:[93m73[0m:[93m7[0m - [91merror[0m[90m TS2353: [0mObject literal may only specify known properties, and 'f1' does not exist in type '{ kind: "Ctor8"; f0: _uniq_1; }'.

[7m73[0m       f1: { kind: "Ctor8", f0: { kind: "Ctor5" }, f1: { kind: "Ctor7" } },
[7m  [0m [91m      ~~[0m
[96mstarcoderbase-7b/multitok/prog_15/var_8.ts[0m:[93m85[0m:[93m7[0m - [91merror[0m[90m TS2353: [0mObject literal may only specify known properties, and 'f1' does not exist in type '{ kind: "Ctor8"; f0: _uniq_1; }'.

[7m85[0m       f1: {
[7m  [0m [91m      ~~[0m
[96mstarcoderbase-7b/multitok/prog_15/var_8.ts[0m:[93m97[0m:[93m7[0m - [91merror[0m[90m TS2353: [0mObject literal may only specify known properties, and 'f1' does not exist in type '{ kind: "Ctor8"; f0: _uniq_1; }'.

[7m97[0m       f1: {
[7m  [0m [91m      ~~[0m
[96mstarcoderbase-7b/multitok/prog_15/var_8.ts[0m:[93m109[0m:[93m7[0m - [91merror[0m[90m TS2353: [0mObject literal may only specify known properties, and 'f1' does not exist in type '{ kind: "Ctor8"; f0: _uniq_1; }'.

[7m109[0m       f1: {
[7m   [0m [91m      ~~[0m
[96mstarcoderbase-7b/multitok/prog_15/var_8.ts[0m:[93m121[0m:[93m7[0m - [91merror[0m[90m TS2353: [0mObject literal may only specify known properties, and 'f1' does not exist in type '{ kind: "Ctor8"; f0: _uniq_1; }'.

[7m121[0m       f1: {
[7m   [0m [91m      ~~[0m
[96mstarcoderbase-7b/multitok/prog_15/var_8.ts[0m:[93m5[0m:[93m80[0m - [91merror[0m[90m TS1128: [0mDeclaration or statement expected.

[7m5[0m type _uniq_2 = { kind: "Ctor7" } | { kind: "Ctor8"; f0: _uniq_1 }; f1: _uniq_2 };
[7m [0m [91m                                                                               ~[0m

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
  diagnosticCodes: [
    2693, 2339, 2353,
    2353, 2353, 2353,
    2353, 2353, 2353,
    2353, 2353, 2353,
    1128
  ]
}
