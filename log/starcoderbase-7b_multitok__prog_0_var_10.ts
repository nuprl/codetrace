/home/franlucc/.npm_packages/_npx/1bf7c3c15bf47d04/node_modules/ts-node/src/index.ts:859
    return new TSError(diagnosticText, diagnosticCodes, diagnostics);
           ^
TSError: ⨯ Unable to compile TypeScript:
[96mstarcoderbase-7b/multitok/prog_0/var_10.ts[0m:[93m10[0m:[93m22[0m - [91merror[0m[90m TS7008: [0mMember 'f0' implicitly has an 'any' type.

[7m10[0m   | { kind: "Ctor9"; f0; f1: _uniq_0; f2: _uniq_2 };
[7m  [0m [91m                     ~~[0m
[96mstarcoderbase-7b/multitok/prog_0/var_10.ts[0m:[93m35[0m:[93m26[0m - [91merror[0m[90m TS2322: [0mType '_uniq_1' is not assignable to type '_uniq_0'.
  Type '{ kind: "Ctor5"; }' is not assignable to type '_uniq_0'.
    Property 'f0' is missing in type '{ kind: "Ctor5"; }' but required in type '{ kind: "Ctor4"; f0: _uniq_0; }'.

[7m35[0m         { kind: "Ctor6", f0: _uniq_17, f1: _uniq_10(_uniq_16) },
[7m  [0m [91m                         ~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_10.ts[0m:[93m3[0m:[93m53[0m
    [7m3[0m type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4"; f0: _uniq_0 };
    [7m [0m [96m                                                    ~~[0m
    'f0' is declared here.
[96mstarcoderbase-7b/multitok/prog_0/var_10.ts[0m:[93m35[0m:[93m53[0m - [91merror[0m[90m TS2345: [0mArgument of type '_uniq_0' is not assignable to parameter of type '_uniq_2'.
  Type '{ kind: "Ctor3"; }' is not assignable to type '_uniq_2'.
    Type '{ kind: "Ctor3"; }' is missing the following properties from type '{ kind: "Ctor8"; f0: _uniq_0; f1: _uniq_1; f2: _uniq_2; }': f0, f1, f2

[7m35[0m         { kind: "Ctor6", f0: _uniq_17, f1: _uniq_10(_uniq_16) },
[7m  [0m [91m                                                    ~~~~~~~~[0m
[96mstarcoderbase-7b/multitok/prog_0/var_10.ts[0m:[93m48[0m:[93m9[0m - [91merror[0m[90m TS2820: [0mType '"Ctor8"' is not assignable to type '"Ctor3" | "Ctor4"'. Did you mean '"Ctor3"'?

[7m48[0m         kind: "Ctor8",
[7m  [0m [91m        ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_10.ts[0m:[93m3[0m:[93m18[0m
    [7m3[0m type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4"; f0: _uniq_0 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_0'
[96mstarcoderbase-7b/multitok/prog_0/var_10.ts[0m:[93m49[0m:[93m15[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor3" | "Ctor4"'. Did you mean '"Ctor3"'?

[7m49[0m         f0: { kind: "Ctor7" },
[7m  [0m [91m              ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_10.ts[0m:[93m3[0m:[93m18[0m
    [7m3[0m type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4"; f0: _uniq_0 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_0'
[96mstarcoderbase-7b/multitok/prog_0/var_10.ts[0m:[93m53[0m:[93m13[0m - [91merror[0m[90m TS2820: [0mType '"Ctor4"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m53[0m       f1: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
[7m  [0m [91m            ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_10.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_0; f1: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_0/var_10.ts[0m:[93m69[0m:[93m13[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor3" | "Ctor4"'. Did you mean '"Ctor3"'?

[7m69[0m       f0: { kind: "Ctor7" },
[7m  [0m [91m            ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_10.ts[0m:[93m3[0m:[93m18[0m
    [7m3[0m type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4"; f0: _uniq_0 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_0'
[96mstarcoderbase-7b/multitok/prog_0/var_10.ts[0m:[93m70[0m:[93m13[0m - [91merror[0m[90m TS2820: [0mType '"Ctor4"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m70[0m       f1: { kind: "Ctor4", f0: { kind: "Ctor3" } },
[7m  [0m [91m            ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_10.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_0; f1: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_0/var_10.ts[0m:[93m73[0m:[93m15[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor3" | "Ctor4"'. Did you mean '"Ctor3"'?

[7m73[0m         f0: { kind: "Ctor7" },
[7m  [0m [91m              ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_10.ts[0m:[93m3[0m:[93m18[0m
    [7m3[0m type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4"; f0: _uniq_0 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_0'
[96mstarcoderbase-7b/multitok/prog_0/var_10.ts[0m:[93m74[0m:[93m15[0m - [91merror[0m[90m TS2820: [0mType '"Ctor4"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m74[0m         f1: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
[7m  [0m [91m              ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_10.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_0; f1: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_0/var_10.ts[0m:[93m96[0m:[93m9[0m - [91merror[0m[90m TS2820: [0mType '"Ctor8"' is not assignable to type '"Ctor3" | "Ctor4"'. Did you mean '"Ctor3"'?

[7m96[0m         kind: "Ctor8",
[7m  [0m [91m        ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_10.ts[0m:[93m3[0m:[93m18[0m
    [7m3[0m type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4"; f0: _uniq_0 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_0'
[96mstarcoderbase-7b/multitok/prog_0/var_10.ts[0m:[93m98[0m:[93m11[0m - [91merror[0m[90m TS2820: [0mType '"Ctor8"' is not assignable to type '"Ctor3" | "Ctor4"'. Did you mean '"Ctor3"'?

[7m98[0m           kind: "Ctor8",
[7m  [0m [91m          ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_10.ts[0m:[93m3[0m:[93m18[0m
    [7m3[0m type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4"; f0: _uniq_0 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_0'
[96mstarcoderbase-7b/multitok/prog_0/var_10.ts[0m:[93m99[0m:[93m17[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor3" | "Ctor4"'. Did you mean '"Ctor3"'?

[7m99[0m           f0: { kind: "Ctor7" },
[7m  [0m [91m                ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_10.ts[0m:[93m3[0m:[93m18[0m
    [7m3[0m type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4"; f0: _uniq_0 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_0'
[96mstarcoderbase-7b/multitok/prog_0/var_10.ts[0m:[93m112[0m:[93m9[0m - [91merror[0m[90m TS2820: [0mType '"Ctor4"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m112[0m         kind: "Ctor4",
[7m   [0m [91m        ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_10.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_0; f1: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_0/var_10.ts[0m:[93m118[0m:[93m11[0m - [91merror[0m[90m TS2820: [0mType '"Ctor8"' is not assignable to type '"Ctor3" | "Ctor4"'. Did you mean '"Ctor3"'?

[7m118[0m           kind: "Ctor8",
[7m   [0m [91m          ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_10.ts[0m:[93m3[0m:[93m18[0m
    [7m3[0m type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4"; f0: _uniq_0 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_0'
[96mstarcoderbase-7b/multitok/prog_0/var_10.ts[0m:[93m119[0m:[93m17[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor3" | "Ctor4"'. Did you mean '"Ctor3"'?

[7m119[0m           f0: { kind: "Ctor7" },
[7m   [0m [91m                ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_10.ts[0m:[93m3[0m:[93m18[0m
    [7m3[0m type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4"; f0: _uniq_0 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_0'
[96mstarcoderbase-7b/multitok/prog_0/var_10.ts[0m:[93m133[0m:[93m11[0m - [91merror[0m[90m TS2820: [0mType '"Ctor4"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m133[0m           kind: "Ctor4",
[7m   [0m [91m          ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_10.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_0; f1: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_0/var_10.ts[0m:[93m147[0m:[93m17[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor3" | "Ctor4"'. Did you mean '"Ctor3"'?

[7m147[0m           f0: { kind: "Ctor7" },
[7m   [0m [91m                ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_10.ts[0m:[93m3[0m:[93m18[0m
    [7m3[0m type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4"; f0: _uniq_0 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_0'
[96mstarcoderbase-7b/multitok/prog_0/var_10.ts[0m:[93m149[0m:[93m13[0m - [91merror[0m[90m TS2820: [0mType '"Ctor4"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m149[0m             kind: "Ctor4",
[7m   [0m [91m            ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_10.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_0; f1: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_0/var_10.ts[0m:[93m240[0m:[93m13[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor3" | "Ctor4"'. Did you mean '"Ctor3"'?

[7m240[0m       f0: { kind: "Ctor7" },
[7m   [0m [91m            ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_10.ts[0m:[93m3[0m:[93m18[0m
    [7m3[0m type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4"; f0: _uniq_0 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_0'
[96mstarcoderbase-7b/multitok/prog_0/var_10.ts[0m:[93m241[0m:[93m13[0m - [91merror[0m[90m TS2820: [0mType '"Ctor3"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m241[0m       f1: { kind: "Ctor3" },
[7m   [0m [91m            ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_10.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_0; f1: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_0/var_10.ts[0m:[93m244[0m:[93m15[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor3" | "Ctor4"'. Did you mean '"Ctor3"'?

[7m244[0m         f0: { kind: "Ctor7" },
[7m   [0m [91m              ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_10.ts[0m:[93m3[0m:[93m18[0m
    [7m3[0m type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4"; f0: _uniq_0 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_0'
[96mstarcoderbase-7b/multitok/prog_0/var_10.ts[0m:[93m245[0m:[93m15[0m - [91merror[0m[90m TS2820: [0mType '"Ctor4"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m245[0m         f1: { kind: "Ctor4", f0: { kind: "Ctor3" } },
[7m   [0m [91m              ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_10.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_0; f1: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_0/var_10.ts[0m:[93m248[0m:[93m17[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor3" | "Ctor4"'. Did you mean '"Ctor3"'?

[7m248[0m           f0: { kind: "Ctor7" },
[7m   [0m [91m                ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_10.ts[0m:[93m3[0m:[93m18[0m
    [7m3[0m type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4"; f0: _uniq_0 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_0'
[96mstarcoderbase-7b/multitok/prog_0/var_10.ts[0m:[93m249[0m:[93m17[0m - [91merror[0m[90m TS2820: [0mType '"Ctor4"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m249[0m           f1: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
[7m   [0m [91m                ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_10.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_0; f1: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_0/var_10.ts[0m:[93m272[0m:[93m9[0m - [91merror[0m[90m TS2820: [0mType '"Ctor8"' is not assignable to type '"Ctor3" | "Ctor4"'. Did you mean '"Ctor3"'?

[7m272[0m         kind: "Ctor8",
[7m   [0m [91m        ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_10.ts[0m:[93m3[0m:[93m18[0m
    [7m3[0m type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4"; f0: _uniq_0 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_0'
[96mstarcoderbase-7b/multitok/prog_0/var_10.ts[0m:[93m274[0m:[93m11[0m - [91merror[0m[90m TS2820: [0mType '"Ctor8"' is not assignable to type '"Ctor3" | "Ctor4"'. Did you mean '"Ctor3"'?

[7m274[0m           kind: "Ctor8",
[7m   [0m [91m          ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_10.ts[0m:[93m3[0m:[93m18[0m
    [7m3[0m type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4"; f0: _uniq_0 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_0'
[96mstarcoderbase-7b/multitok/prog_0/var_10.ts[0m:[93m275[0m:[93m17[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor3" | "Ctor4"'. Did you mean '"Ctor3"'?

[7m275[0m           f0: { kind: "Ctor7" },
[7m   [0m [91m                ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_10.ts[0m:[93m3[0m:[93m18[0m
    [7m3[0m type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4"; f0: _uniq_0 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_0'
[96mstarcoderbase-7b/multitok/prog_0/var_10.ts[0m:[93m282[0m:[93m13[0m - [91merror[0m[90m TS2820: [0mType '"Ctor3"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m282[0m       f1: { kind: "Ctor3" },
[7m   [0m [91m            ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_10.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_0; f1: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_0/var_10.ts[0m:[93m285[0m:[93m15[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor3" | "Ctor4"'. Did you mean '"Ctor3"'?

[7m285[0m         f0: { kind: "Ctor7" },
[7m   [0m [91m              ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_10.ts[0m:[93m3[0m:[93m18[0m
    [7m3[0m type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4"; f0: _uniq_0 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_0'
[96mstarcoderbase-7b/multitok/prog_0/var_10.ts[0m:[93m286[0m:[93m15[0m - [91merror[0m[90m TS2820: [0mType '"Ctor4"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m286[0m         f1: { kind: "Ctor4", f0: { kind: "Ctor3" } },
[7m   [0m [91m              ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_10.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_0; f1: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_0/var_10.ts[0m:[93m289[0m:[93m17[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor3" | "Ctor4"'. Did you mean '"Ctor3"'?

[7m289[0m           f0: { kind: "Ctor7" },
[7m   [0m [91m                ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_10.ts[0m:[93m3[0m:[93m18[0m
    [7m3[0m type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4"; f0: _uniq_0 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_0'
[96mstarcoderbase-7b/multitok/prog_0/var_10.ts[0m:[93m290[0m:[93m17[0m - [91merror[0m[90m TS2820: [0mType '"Ctor4"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m290[0m           f1: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
[7m   [0m [91m                ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_10.ts[0m:[93m4[0m:[93m18[0m
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
    7008, 2322, 2345, 2820, 2820,
    2820, 2820, 2820, 2820, 2820,
    2820, 2820, 2820, 2820, 2820,
    2820, 2820, 2820, 2820, 2820,
    2820, 2820, 2820, 2820, 2820,
    2820, 2820, 2820, 2820, 2820,
    2820, 2820, 2820
  ]
}
