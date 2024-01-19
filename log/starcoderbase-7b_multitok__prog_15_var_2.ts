/home/franlucc/.npm_packages/_npx/1bf7c3c15bf47d04/node_modules/ts-node/src/index.ts:859
    return new TSError(diagnosticText, diagnosticCodes, diagnostics);
           ^
TSError: ⨯ Unable to compile TypeScript:
[96mstarcoderbase-7b/multitok/prog_15/var_2.ts[0m:[93m3[0m:[93m59[0m - [91merror[0m[90m TS2693: [0m'_uniq_0' only refers to a type, but is being used as a value here.

[7m3[0m type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4" }; f0: _uniq_0 };
[7m [0m [91m                                                          ~~~~~~~[0m
[96mstarcoderbase-7b/multitok/prog_15/var_2.ts[0m:[93m3[0m:[93m67[0m - [91merror[0m[90m TS1128: [0mDeclaration or statement expected.

[7m3[0m type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4" }; f0: _uniq_0 };
[7m [0m [91m                                                                  ~[0m

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
  diagnosticCodes: [ 2693, 1128 ]
}
