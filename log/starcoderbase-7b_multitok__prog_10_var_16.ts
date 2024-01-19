/home/franlucc/.npm_packages/_npx/1bf7c3c15bf47d04/node_modules/ts-node/src/index.ts:859
    return new TSError(diagnosticText, diagnosticCodes, diagnostics);
           ^
TSError: ⨯ Unable to compile TypeScript:
[96mstarcoderbase-7b/multitok/prog_10/var_16.ts[0m:[93m24[0m:[93m16[0m - [91merror[0m[90m TS2820: [0mType '"Ctor5"' is not assignable to type '"Ctor7" | "Ctor8"'. Did you mean '"Ctor7"'?

[7m24[0m       return { kind: "Ctor5" };
[7m  [0m [91m               ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_16.ts[0m:[93m6[0m:[93m7[0m
    [7m6[0m   | { kind: "Ctor7" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_2'
[96mstarcoderbase-7b/multitok/prog_10/var_16.ts[0m:[93m32[0m:[93m20[0m - [91merror[0m[90m TS2820: [0mType '"Ctor6"' is not assignable to type '"Ctor7" | "Ctor8"'. Did you mean '"Ctor7"'?

[7m32[0m           return { kind: "Ctor6", f0: { kind: "Ctor5" } };
[7m  [0m [91m                   ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_16.ts[0m:[93m6[0m:[93m7[0m
    [7m6[0m   | { kind: "Ctor7" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_2'
[96mstarcoderbase-7b/multitok/prog_10/var_16.ts[0m:[93m32[0m:[93m41[0m - [91merror[0m[90m TS2820: [0mType '"Ctor5"' is not assignable to type '"Ctor7" | "Ctor8"'. Did you mean '"Ctor7"'?

[7m32[0m           return { kind: "Ctor6", f0: { kind: "Ctor5" } };
[7m  [0m [91m                                        ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_16.ts[0m:[93m6[0m:[93m7[0m
    [7m6[0m   | { kind: "Ctor7" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_2'
[96mstarcoderbase-7b/multitok/prog_10/var_16.ts[0m:[93m36[0m:[93m11[0m - [91merror[0m[90m TS2322: [0mType '_uniq_1' is not assignable to type '_uniq_2'.
  Type '{ kind: "Ctor5"; }' is not assignable to type '_uniq_2'.
    Type '{ kind: "Ctor5"; }' is missing the following properties from type '{ kind: "Ctor8"; f0: _uniq_2; f1: _uniq_0; f2: _uniq_2; }': f0, f1, f2

[7m36[0m           return _uniq_9(
[7m  [0m [91m          ~~~~~~[0m
[96mstarcoderbase-7b/multitok/prog_10/var_16.ts[0m:[93m37[0m:[93m13[0m - [91merror[0m[90m TS2345: [0mArgument of type '_uniq_2' is not assignable to parameter of type '_uniq_1'.
  Type '{ kind: "Ctor7"; }' is not assignable to type '_uniq_1'.
    Property 'f0' is missing in type '{ kind: "Ctor7"; }' but required in type '{ kind: "Ctor6"; f0: _uniq_1; }'.

[7m37[0m             _uniq_10(_uniq_16, _uniq_19),
[7m  [0m [91m            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_16.ts[0m:[93m4[0m:[93m53[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_1 };
    [7m [0m [96m                                                    ~~[0m
    'f0' is declared here.

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
  diagnosticCodes: [ 2820, 2820, 2820, 2322, 2345 ]
}
