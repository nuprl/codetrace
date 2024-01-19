/home/franlucc/.npm_packages/_npx/1bf7c3c15bf47d04/node_modules/ts-node/src/index.ts:859
    return new TSError(diagnosticText, diagnosticCodes, diagnostics);
           ^
TSError: ⨯ Unable to compile TypeScript:
[96mstarcoderbase-7b/multitok/prog_14/var_3.ts[0m:[93m7[0m:[93m6[0m - [91merror[0m[90m TS2300: [0mDuplicate identifier '_uniq_2'.

[7m7[0m type _uniq_2 = { kind: " }
[7m [0m [91m     ~~~~~~~[0m
[96mstarcoderbase-7b/multitok/prog_14/var_3.ts[0m:[93m9[0m:[93m6[0m - [91merror[0m[90m TS2300: [0mDuplicate identifier '_uniq_2'.

[7m9[0m type _uniq_2 = { kind: "Ctor7" } | { kind: "Ctor8"; f0: _uniq_2 };
[7m [0m [91m     ~~~~~~~[0m
[96mstarcoderbase-7b/multitok/prog_14/var_3.ts[0m:[93m11[0m:[93m57[0m - [91merror[0m[90m TS2366: [0mFunction lacks ending return statement and return type does not include 'undefined'.

[7m11[0m function _uniq_9(_uniq_11: _uniq_2, _uniq_12: _uniq_2): _uniq_2 {
[7m  [0m [91m                                                        ~~~~~~~[0m
[96mstarcoderbase-7b/multitok/prog_14/var_3.ts[0m:[93m13[0m:[93m10[0m - [91merror[0m[90m TS2678: [0mType '"Ctor7"' is not comparable to type '{ kind: "Ctor6"; f0: _uniq_1; f1: _uniq_0; f2: _uniq_1; } | " }"'.

[7m13[0m     case "Ctor7": {
[7m  [0m [91m         ~~~~~~~[0m
[96mstarcoderbase-7b/multitok/prog_14/var_3.ts[0m:[93m16[0m:[93m10[0m - [91merror[0m[90m TS2678: [0mType '"Ctor8"' is not comparable to type '{ kind: "Ctor6"; f0: _uniq_1; f1: _uniq_0; f2: _uniq_1; } | " }"'.

[7m16[0m     case "Ctor8": {
[7m  [0m [91m         ~~~~~~~[0m
[96mstarcoderbase-7b/multitok/prog_14/var_3.ts[0m:[93m17[0m:[93m31[0m - [91merror[0m[90m TS2339: [0mProperty 'f0' does not exist on type '_uniq_2'.

[7m17[0m       let _uniq_13 = _uniq_11.f0;
[7m  [0m [91m                              ~~[0m
[96mstarcoderbase-7b/multitok/prog_14/var_3.ts[0m:[93m18[0m:[93m16[0m - [91merror[0m[90m TS2322: [0mType '"Ctor8"' is not assignable to type '{ kind: "Ctor6"; f0: _uniq_1; f1: _uniq_0; f2: _uniq_1; } | " }"'.

[7m18[0m       return { kind: "Ctor8", f0: _uniq_9(_uniq_13, _uniq_12) };
[7m  [0m [91m               ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_14/var_3.ts[0m:[93m7[0m:[93m18[0m
    [7m7[0m type _uniq_2 = { kind: " }
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_2'
[96mstarcoderbase-7b/multitok/prog_14/var_3.ts[0m:[93m26[0m:[93m16[0m - [91merror[0m[90m TS2322: [0mType '"Ctor8"' is not assignable to type '{ kind: "Ctor6"; f0: _uniq_1; f1: _uniq_0; f2: _uniq_1; } | " }"'.

[7m26[0m       return { kind: "Ctor8", f0: { kind: "Ctor7" } };
[7m  [0m [91m               ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_14/var_3.ts[0m:[93m7[0m:[93m18[0m
    [7m7[0m type _uniq_2 = { kind: " }
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_2'
[96mstarcoderbase-7b/multitok/prog_14/var_3.ts[0m:[93m7[0m:[93m27[0m - [91merror[0m[90m TS1002: [0mUnterminated string literal.

[7m7[0m type _uniq_2 = { kind: " }
[7m [0m [91m                          [0m
[96mstarcoderbase-7b/multitok/prog_14/var_3.ts[0m:[93m9[0m:[93m1[0m - [91merror[0m[90m TS1131: [0mProperty or signature expected.

[7m9[0m type _uniq_2 = { kind: "Ctor7" } | { kind: "Ctor8"; f0: _uniq_2 };
[7m [0m [91m~~~~[0m

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
    2300, 2300, 2366,
    2678, 2678, 2339,
    2322, 2322, 1002,
    1131
  ]
}
