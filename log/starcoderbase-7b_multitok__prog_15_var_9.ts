/home/franlucc/.npm_packages/_npx/1bf7c3c15bf47d04/node_modules/ts-node/src/index.ts:859
    return new TSError(diagnosticText, diagnosticCodes, diagnostics);
           ^
TSError: ⨯ Unable to compile TypeScript:
[96mstarcoderbase-7b/multitok/prog_15/var_9.ts[0m:[93m17[0m:[93m39[0m - [91merror[0m[90m TS2345: [0mArgument of type '_uniq_0' is not assignable to parameter of type '_uniq_2'.
  Type '{ kind: "Ctor3"; }' is not assignable to type '_uniq_2'.
    Type '{ kind: "Ctor3"; }' is missing the following properties from type '{ kind: "Ctor8"; f0: _uniq_1; f1: _uniq_0; }': f0, f1

[7m17[0m       let _uniq_13: _uniq_1 = _uniq_9(_uniq_12);
[7m  [0m [91m                                      ~~~~~~~~[0m
[96mstarcoderbase-7b/multitok/prog_15/var_9.ts[0m:[93m40[0m:[93m59[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor3" | "Ctor4"'. Did you mean '"Ctor3"'?

[7m40[0m     _uniq_9({ kind: "Ctor8", f0: { kind: "Ctor6" }, f1: { kind: "Ctor7" } }),
[7m  [0m [91m                                                          ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_15/var_9.ts[0m:[93m3[0m:[93m18[0m
    [7m3[0m type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4"; f0: _uniq_0 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_0'
[96mstarcoderbase-7b/multitok/prog_15/var_9.ts[0m:[93m44[0m:[93m59[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor3" | "Ctor4"'. Did you mean '"Ctor3"'?

[7m44[0m     _uniq_9({ kind: "Ctor8", f0: { kind: "Ctor5" }, f1: { kind: "Ctor7" } }),
[7m  [0m [91m                                                          ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_15/var_9.ts[0m:[93m3[0m:[93m18[0m
    [7m3[0m type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4"; f0: _uniq_0 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_0'
[96mstarcoderbase-7b/multitok/prog_15/var_9.ts[0m:[93m51[0m:[93m13[0m - [91merror[0m[90m TS2820: [0mType '"Ctor8"' is not assignable to type '"Ctor3" | "Ctor4"'. Did you mean '"Ctor3"'?

[7m51[0m       f1: { kind: "Ctor8", f0: { kind: "Ctor6" }, f1: { kind: "Ctor7" } },
[7m  [0m [91m            ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_15/var_9.ts[0m:[93m3[0m:[93m18[0m
    [7m3[0m type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4"; f0: _uniq_0 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_0'
[96mstarcoderbase-7b/multitok/prog_15/var_9.ts[0m:[93m51[0m:[93m34[0m - [91merror[0m[90m TS2820: [0mType '"Ctor6"' is not assignable to type '"Ctor3" | "Ctor4"'. Did you mean '"Ctor3"'?

[7m51[0m       f1: { kind: "Ctor8", f0: { kind: "Ctor6" }, f1: { kind: "Ctor7" } },
[7m  [0m [91m                                 ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_15/var_9.ts[0m:[93m3[0m:[93m18[0m
    [7m3[0m type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4"; f0: _uniq_0 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_0'
[96mstarcoderbase-7b/multitok/prog_15/var_9.ts[0m:[93m59[0m:[93m13[0m - [91merror[0m[90m TS2820: [0mType '"Ctor8"' is not assignable to type '"Ctor3" | "Ctor4"'. Did you mean '"Ctor3"'?

[7m59[0m       f1: { kind: "Ctor8", f0: { kind: "Ctor5" }, f1: { kind: "Ctor7" } },
[7m  [0m [91m            ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_15/var_9.ts[0m:[93m3[0m:[93m18[0m
    [7m3[0m type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4"; f0: _uniq_0 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_0'
[96mstarcoderbase-7b/multitok/prog_15/var_9.ts[0m:[93m59[0m:[93m34[0m - [91merror[0m[90m TS2820: [0mType '"Ctor5"' is not assignable to type '"Ctor3" | "Ctor4"'. Did you mean '"Ctor3"'?

[7m59[0m       f1: { kind: "Ctor8", f0: { kind: "Ctor5" }, f1: { kind: "Ctor7" } },
[7m  [0m [91m                                 ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_15/var_9.ts[0m:[93m3[0m:[93m18[0m
    [7m3[0m type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4"; f0: _uniq_0 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_0'
[96mstarcoderbase-7b/multitok/prog_15/var_9.ts[0m:[93m67[0m:[93m13[0m - [91merror[0m[90m TS2820: [0mType '"Ctor8"' is not assignable to type '"Ctor3" | "Ctor4"'. Did you mean '"Ctor3"'?

[7m67[0m       f1: { kind: "Ctor8", f0: { kind: "Ctor6" }, f1: { kind: "Ctor7" } },
[7m  [0m [91m            ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_15/var_9.ts[0m:[93m3[0m:[93m18[0m
    [7m3[0m type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4"; f0: _uniq_0 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_0'
[96mstarcoderbase-7b/multitok/prog_15/var_9.ts[0m:[93m67[0m:[93m34[0m - [91merror[0m[90m TS2820: [0mType '"Ctor6"' is not assignable to type '"Ctor3" | "Ctor4"'. Did you mean '"Ctor3"'?

[7m67[0m       f1: { kind: "Ctor8", f0: { kind: "Ctor6" }, f1: { kind: "Ctor7" } },
[7m  [0m [91m                                 ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_15/var_9.ts[0m:[93m3[0m:[93m18[0m
    [7m3[0m type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4"; f0: _uniq_0 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_0'
[96mstarcoderbase-7b/multitok/prog_15/var_9.ts[0m:[93m75[0m:[93m13[0m - [91merror[0m[90m TS2820: [0mType '"Ctor8"' is not assignable to type '"Ctor3" | "Ctor4"'. Did you mean '"Ctor3"'?

[7m75[0m       f1: { kind: "Ctor8", f0: { kind: "Ctor5" }, f1: { kind: "Ctor7" } },
[7m  [0m [91m            ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_15/var_9.ts[0m:[93m3[0m:[93m18[0m
    [7m3[0m type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4"; f0: _uniq_0 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_0'
[96mstarcoderbase-7b/multitok/prog_15/var_9.ts[0m:[93m75[0m:[93m34[0m - [91merror[0m[90m TS2820: [0mType '"Ctor5"' is not assignable to type '"Ctor3" | "Ctor4"'. Did you mean '"Ctor3"'?

[7m75[0m       f1: { kind: "Ctor8", f0: { kind: "Ctor5" }, f1: { kind: "Ctor7" } },
[7m  [0m [91m                                 ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_15/var_9.ts[0m:[93m3[0m:[93m18[0m
    [7m3[0m type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4"; f0: _uniq_0 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_0'
[96mstarcoderbase-7b/multitok/prog_15/var_9.ts[0m:[93m88[0m:[93m9[0m - [91merror[0m[90m TS2820: [0mType '"Ctor8"' is not assignable to type '"Ctor3" | "Ctor4"'. Did you mean '"Ctor3"'?

[7m88[0m         kind: "Ctor8",
[7m  [0m [91m        ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_15/var_9.ts[0m:[93m3[0m:[93m18[0m
    [7m3[0m type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4"; f0: _uniq_0 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_0'
[96mstarcoderbase-7b/multitok/prog_15/var_9.ts[0m:[93m89[0m:[93m15[0m - [91merror[0m[90m TS2820: [0mType '"Ctor6"' is not assignable to type '"Ctor3" | "Ctor4"'. Did you mean '"Ctor3"'?

[7m89[0m         f0: { kind: "Ctor6" },
[7m  [0m [91m              ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_15/var_9.ts[0m:[93m3[0m:[93m18[0m
    [7m3[0m type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4"; f0: _uniq_0 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_0'
[96mstarcoderbase-7b/multitok/prog_15/var_9.ts[0m:[93m100[0m:[93m9[0m - [91merror[0m[90m TS2820: [0mType '"Ctor8"' is not assignable to type '"Ctor3" | "Ctor4"'. Did you mean '"Ctor3"'?

[7m100[0m         kind: "Ctor8",
[7m   [0m [91m        ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_15/var_9.ts[0m:[93m3[0m:[93m18[0m
    [7m3[0m type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4"; f0: _uniq_0 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_0'
[96mstarcoderbase-7b/multitok/prog_15/var_9.ts[0m:[93m101[0m:[93m15[0m - [91merror[0m[90m TS2820: [0mType '"Ctor5"' is not assignable to type '"Ctor3" | "Ctor4"'. Did you mean '"Ctor3"'?

[7m101[0m         f0: { kind: "Ctor5" },
[7m   [0m [91m              ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_15/var_9.ts[0m:[93m3[0m:[93m18[0m
    [7m3[0m type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4"; f0: _uniq_0 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_0'
[96mstarcoderbase-7b/multitok/prog_15/var_9.ts[0m:[93m112[0m:[93m9[0m - [91merror[0m[90m TS2820: [0mType '"Ctor8"' is not assignable to type '"Ctor3" | "Ctor4"'. Did you mean '"Ctor3"'?

[7m112[0m         kind: "Ctor8",
[7m   [0m [91m        ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_15/var_9.ts[0m:[93m3[0m:[93m18[0m
    [7m3[0m type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4"; f0: _uniq_0 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_0'
[96mstarcoderbase-7b/multitok/prog_15/var_9.ts[0m:[93m113[0m:[93m15[0m - [91merror[0m[90m TS2820: [0mType '"Ctor6"' is not assignable to type '"Ctor3" | "Ctor4"'. Did you mean '"Ctor3"'?

[7m113[0m         f0: { kind: "Ctor6" },
[7m   [0m [91m              ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_15/var_9.ts[0m:[93m3[0m:[93m18[0m
    [7m3[0m type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4"; f0: _uniq_0 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_0'
[96mstarcoderbase-7b/multitok/prog_15/var_9.ts[0m:[93m124[0m:[93m9[0m - [91merror[0m[90m TS2820: [0mType '"Ctor8"' is not assignable to type '"Ctor3" | "Ctor4"'. Did you mean '"Ctor3"'?

[7m124[0m         kind: "Ctor8",
[7m   [0m [91m        ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_15/var_9.ts[0m:[93m3[0m:[93m18[0m
    [7m3[0m type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4"; f0: _uniq_0 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_0'
[96mstarcoderbase-7b/multitok/prog_15/var_9.ts[0m:[93m125[0m:[93m15[0m - [91merror[0m[90m TS2820: [0mType '"Ctor5"' is not assignable to type '"Ctor3" | "Ctor4"'. Did you mean '"Ctor3"'?

[7m125[0m         f0: { kind: "Ctor5" },
[7m   [0m [91m              ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_15/var_9.ts[0m:[93m3[0m:[93m18[0m
    [7m3[0m type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4"; f0: _uniq_0 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_0'
[96mstarcoderbase-7b/multitok/prog_15/var_9.ts[0m:[93m7[0m:[93m32[0m - [91merror[0m[90m TS1002: [0mUnterminated string literal.

[7m7[0m type _uniq_4 = { kind: "Ctor };
[7m [0m [91m                               [0m
[96mstarcoderbase-7b/multitok/prog_15/var_9.ts[0m:[93m9[0m:[93m1[0m - [91merror[0m[90m TS1131: [0mProperty or signature expected.

[7m9[0m function _uniq_9(_uniq_10: _uniq_2): _uniq_1 {
[7m [0m [91m~~~~~~~~[0m

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
    2345, 2820, 2820, 2820,
    2820, 2820, 2820, 2820,
    2820, 2820, 2820, 2820,
    2820, 2820, 2820, 2820,
    2820, 2820, 2820, 1002,
    1131
  ]
}
