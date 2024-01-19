/home/franlucc/.npm_packages/_npx/1bf7c3c15bf47d04/node_modules/ts-node/src/index.ts:859
    return new TSError(diagnosticText, diagnosticCodes, diagnostics);
           ^
TSError: ⨯ Unable to compile TypeScript:
[96mstarcoderbase-7b/multitok/prog_0/var_12.ts[0m:[93m10[0m:[93m49[0m - [91merror[0m[90m TS7008: [0mMember 'f' implicitly has an 'any' type.

[7m10[0m   | { kind: "Ctor10"; f0: _uniq_3; f1: _uniq_2; f };
[7m  [0m [91m                                                ~[0m
[96mstarcoderbase-7b/multitok/prog_0/var_12.ts[0m:[93m36[0m:[93m18[0m - [91merror[0m[90m TS2345: [0mArgument of type '_uniq_1' is not assignable to parameter of type '_uniq_2'.
  Type '{ kind: "Ctor5"; }' is not assignable to type '_uniq_2'.
    Type '{ kind: "Ctor5"; }' is missing the following properties from type '{ kind: "Ctor8"; f0: _uniq_2; f1: _uniq_0; f2: _uniq_1; }': f0, f1, f2

[7m36[0m         _uniq_10(_uniq_18)
[7m  [0m [91m                 ~~~~~~~~[0m
[96mstarcoderbase-7b/multitok/prog_0/var_12.ts[0m:[93m51[0m:[93m15[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m51[0m         f2: { kind: "Ctor7" },
[7m  [0m [91m              ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_12.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_0; f1: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_0/var_12.ts[0m:[93m54[0m:[93m13[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m54[0m       f2: { kind: "Ctor7" },
[7m  [0m [91m            ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_12.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_0; f1: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_0/var_12.ts[0m:[93m72[0m:[93m9[0m - [91merror[0m[90m TS2820: [0mType '"Ctor8"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m72[0m         kind: "Ctor8",
[7m  [0m [91m        ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_12.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_0; f1: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_0/var_12.ts[0m:[93m73[0m:[93m15[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor3" | "Ctor4"'. Did you mean '"Ctor3"'?

[7m73[0m         f0: { kind: "Ctor7" },
[7m  [0m [91m              ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_12.ts[0m:[93m3[0m:[93m18[0m
    [7m3[0m type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4"; f0: _uniq_0 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_0'
[96mstarcoderbase-7b/multitok/prog_0/var_12.ts[0m:[93m74[0m:[93m15[0m - [91merror[0m[90m TS2820: [0mType '"Ctor4"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m74[0m         f1: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
[7m  [0m [91m              ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_12.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_0; f1: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_0/var_12.ts[0m:[93m101[0m:[93m17[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m101[0m           f2: { kind: "Ctor7" },
[7m   [0m [91m                ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_12.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_0; f1: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_0/var_12.ts[0m:[93m105[0m:[93m11[0m - [91merror[0m[90m TS2820: [0mType '"Ctor8"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m105[0m           kind: "Ctor8",
[7m   [0m [91m          ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_12.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_0; f1: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_0/var_12.ts[0m:[93m106[0m:[93m17[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor3" | "Ctor4"'. Did you mean '"Ctor3"'?

[7m106[0m           f0: { kind: "Ctor7" },
[7m   [0m [91m                ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_12.ts[0m:[93m3[0m:[93m18[0m
    [7m3[0m type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4"; f0: _uniq_0 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_0'
[96mstarcoderbase-7b/multitok/prog_0/var_12.ts[0m:[93m107[0m:[93m17[0m - [91merror[0m[90m TS2820: [0mType '"Ctor4"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m107[0m           f1: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
[7m   [0m [91m                ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_12.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_0; f1: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_0/var_12.ts[0m:[93m116[0m:[93m9[0m - [91merror[0m[90m TS2820: [0mType '"Ctor8"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m116[0m         kind: "Ctor8",
[7m   [0m [91m        ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_12.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_0; f1: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_0/var_12.ts[0m:[93m118[0m:[93m11[0m - [91merror[0m[90m TS2820: [0mType '"Ctor8"' is not assignable to type '"Ctor3" | "Ctor4"'. Did you mean '"Ctor3"'?

[7m118[0m           kind: "Ctor8",
[7m   [0m [91m          ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_12.ts[0m:[93m3[0m:[93m18[0m
    [7m3[0m type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4"; f0: _uniq_0 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_0'
[96mstarcoderbase-7b/multitok/prog_0/var_12.ts[0m:[93m119[0m:[93m17[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor3" | "Ctor4"'. Did you mean '"Ctor3"'?

[7m119[0m           f0: { kind: "Ctor7" },
[7m   [0m [91m                ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_12.ts[0m:[93m3[0m:[93m18[0m
    [7m3[0m type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4"; f0: _uniq_0 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_0'
[96mstarcoderbase-7b/multitok/prog_0/var_12.ts[0m:[93m133[0m:[93m11[0m - [91merror[0m[90m TS2820: [0mType '"Ctor4"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m133[0m           kind: "Ctor4",
[7m   [0m [91m          ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_12.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_0; f1: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_0/var_12.ts[0m:[93m243[0m:[93m9[0m - [91merror[0m[90m TS2820: [0mType '"Ctor8"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m243[0m         kind: "Ctor8",
[7m   [0m [91m        ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_12.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_0; f1: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_0/var_12.ts[0m:[93m244[0m:[93m15[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor3" | "Ctor4"'. Did you mean '"Ctor3"'?

[7m244[0m         f0: { kind: "Ctor7" },
[7m   [0m [91m              ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_12.ts[0m:[93m3[0m:[93m18[0m
    [7m3[0m type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4"; f0: _uniq_0 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_0'
[96mstarcoderbase-7b/multitok/prog_0/var_12.ts[0m:[93m245[0m:[93m15[0m - [91merror[0m[90m TS2820: [0mType '"Ctor4"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m245[0m         f1: { kind: "Ctor4", f0: { kind: "Ctor3" } },
[7m   [0m [91m              ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_12.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_0; f1: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_0/var_12.ts[0m:[93m277[0m:[93m17[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m277[0m           f2: { kind: "Ctor7" },
[7m   [0m [91m                ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_12.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_0; f1: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_0/var_12.ts[0m:[93m280[0m:[93m15[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m280[0m         f2: { kind: "Ctor7" },
[7m   [0m [91m              ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_12.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_0; f1: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_0/var_12.ts[0m:[93m284[0m:[93m9[0m - [91merror[0m[90m TS2820: [0mType '"Ctor8"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m284[0m         kind: "Ctor8",
[7m   [0m [91m        ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_12.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_0; f1: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_0/var_12.ts[0m:[93m285[0m:[93m15[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor3" | "Ctor4"'. Did you mean '"Ctor3"'?

[7m285[0m         f0: { kind: "Ctor7" },
[7m   [0m [91m              ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_12.ts[0m:[93m3[0m:[93m18[0m
    [7m3[0m type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4"; f0: _uniq_0 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_0'
[96mstarcoderbase-7b/multitok/prog_0/var_12.ts[0m:[93m286[0m:[93m15[0m - [91merror[0m[90m TS2820: [0mType '"Ctor4"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m286[0m         f1: { kind: "Ctor4", f0: { kind: "Ctor3" } },
[7m   [0m [91m              ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_12.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_0; f1: _uniq_1 };
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
    7008, 2345, 2820, 2820,
    2820, 2820, 2820, 2820,
    2820, 2820, 2820, 2820,
    2820, 2820, 2820, 2820,
    2820, 2820, 2820, 2820,
    2820, 2820, 2820
  ]
}
