/home/franlucc/.npm_packages/_npx/1bf7c3c15bf47d04/node_modules/ts-node/src/index.ts:859
    return new TSError(diagnosticText, diagnosticCodes, diagnostics);
           ^
TSError: ⨯ Unable to compile TypeScript:
[96mstarcoderbase-7b/multitok/prog_12/var_8.ts[0m:[93m10[0m:[93m38[0m - [91merror[0m[90m TS2366: [0mFunction lacks ending return statement and return type does not include 'undefined'.

[7m10[0m function _uniq_9(_uniq_10: _uniq_2): _uniq_0 {
[7m  [0m [91m                                     ~~~~~~~[0m
[96mstarcoderbase-7b/multitok/prog_12/var_8.ts[0m:[93m20[0m:[93m14[0m - [91merror[0m[90m TS2678: [0mType '"Ctor7"' is not comparable to type '"Ctor5" | "Ctor6"'.

[7m20[0m         case "Ctor7": {
[7m  [0m [91m             ~~~~~~~[0m
[96mstarcoderbase-7b/multitok/prog_12/var_8.ts[0m:[93m23[0m:[93m14[0m - [91merror[0m[90m TS2678: [0mType '"Ctor8"' is not comparable to type '"Ctor5" | "Ctor6"'.

[7m23[0m         case "Ctor8": {
[7m  [0m [91m             ~~~~~~~[0m
[96mstarcoderbase-7b/multitok/prog_12/var_8.ts[0m:[93m24[0m:[93m35[0m - [91merror[0m[90m TS2339: [0mProperty 'f2' does not exist on type 'never'.

[7m24[0m           let _uniq_16 = _uniq_12.f2;
[7m  [0m [91m                                  ~~[0m
[96mstarcoderbase-7b/multitok/prog_12/var_8.ts[0m:[93m25[0m:[93m35[0m - [91merror[0m[90m TS2339: [0mProperty 'f1' does not exist on type 'never'.

[7m25[0m           let _uniq_15 = _uniq_12.f1;
[7m  [0m [91m                                  ~~[0m
[96mstarcoderbase-7b/multitok/prog_12/var_8.ts[0m:[93m26[0m:[93m35[0m - [91merror[0m[90m TS2339: [0mProperty 'f0' does not exist on type 'never'.

[7m26[0m           let _uniq_14 = _uniq_12.f0;
[7m  [0m [91m                                  ~~[0m
[96mstarcoderbase-7b/multitok/prog_12/var_8.ts[0m:[93m40[0m:[93m13[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m40[0m       f1: { kind: "Ctor7" },
[7m  [0m [91m            ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_12/var_8.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6" };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_12/var_8.ts[0m:[93m41[0m:[93m13[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m41[0m       f2: { kind: "Ctor7" },
[7m  [0m [91m            ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_12/var_8.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6" };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_12/var_8.ts[0m:[93m50[0m:[93m9[0m - [91merror[0m[90m TS2820: [0mType '"Ctor8"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m50[0m         kind: "Ctor8",
[7m  [0m [91m        ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_12/var_8.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6" };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_12/var_8.ts[0m:[93m55[0m:[93m13[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m55[0m       f2: { kind: "Ctor7" },
[7m  [0m [91m            ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_12/var_8.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6" };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_12/var_8.ts[0m:[93m64[0m:[93m9[0m - [91merror[0m[90m TS2820: [0mType '"Ctor8"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m64[0m         kind: "Ctor8",
[7m  [0m [91m        ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_12/var_8.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6" };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_12/var_8.ts[0m:[93m92[0m:[93m13[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m92[0m       f2: { kind: "Ctor7" },
[7m  [0m [91m            ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_12/var_8.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6" };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_12/var_8.ts[0m:[93m101[0m:[93m9[0m - [91merror[0m[90m TS2820: [0mType '"Ctor8"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m101[0m         kind: "Ctor8",
[7m   [0m [91m        ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_12/var_8.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6" };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_12/var_8.ts[0m:[93m129[0m:[93m9[0m - [91merror[0m[90m TS2820: [0mType '"Ctor8"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m129[0m         kind: "Ctor8",
[7m   [0m [91m        ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_12/var_8.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6" };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_12/var_8.ts[0m:[93m145[0m:[93m9[0m - [91merror[0m[90m TS2820: [0mType '"Ctor8"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m145[0m         kind: "Ctor8",
[7m   [0m [91m        ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_12/var_8.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6" };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_12/var_8.ts[0m:[93m190[0m:[93m9[0m - [91merror[0m[90m TS2820: [0mType '"Ctor8"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m190[0m         kind: "Ctor8",
[7m   [0m [91m        ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_12/var_8.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6" };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_12/var_8.ts[0m:[93m209[0m:[93m13[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m209[0m       f1: { kind: "Ctor7" },
[7m   [0m [91m            ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_12/var_8.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6" };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_12/var_8.ts[0m:[93m211[0m:[93m9[0m - [91merror[0m[90m TS2820: [0mType '"Ctor8"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m211[0m         kind: "Ctor8",
[7m   [0m [91m        ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_12/var_8.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6" };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_12/var_8.ts[0m:[93m234[0m:[93m13[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m234[0m       f1: { kind: "Ctor7" },
[7m   [0m [91m            ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_12/var_8.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6" };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_12/var_8.ts[0m:[93m236[0m:[93m9[0m - [91merror[0m[90m TS2820: [0mType '"Ctor8"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m236[0m         kind: "Ctor8",
[7m   [0m [91m        ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_12/var_8.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6" };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_12/var_8.ts[0m:[93m260[0m:[93m9[0m - [91merror[0m[90m TS2820: [0mType '"Ctor8"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m260[0m         kind: "Ctor8",
[7m   [0m [91m        ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_12/var_8.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6" };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_12/var_8.ts[0m:[93m278[0m:[93m13[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m278[0m       f2: { kind: "Ctor7" },
[7m   [0m [91m            ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_12/var_8.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6" };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'

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
    2366, 2678, 2678, 2339,
    2339, 2339, 2820, 2820,
    2820, 2820, 2820, 2820,
    2820, 2820, 2820, 2820,
    2820, 2820, 2820, 2820,
    2820, 2820
  ]
}
