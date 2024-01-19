/home/franlucc/.npm_packages/_npx/1bf7c3c15bf47d04/node_modules/ts-node/src/index.ts:859
    return new TSError(diagnosticText, diagnosticCodes, diagnostics);
           ^
TSError: ⨯ Unable to compile TypeScript:
[96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m22[0m:[93m58[0m - [91merror[0m[90m TS2366: [0mFunction lacks ending return statement and return type does not include 'undefined'.

[7m22[0m function _uniq_10(_uniq_14: _uniq_2, _uniq_15: _uniq_1): _uniq_1 {
[7m  [0m [91m                                                         ~~~~~~~[0m
[96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m38[0m:[93m22[0m - [91merror[0m[90m TS2345: [0mArgument of type '_uniq_0' is not assignable to parameter of type '_uniq_2'.
  Type '{ kind: "Ctor3"; }' is not assignable to type '_uniq_2'.
    Type '{ kind: "Ctor3"; }' is missing the following properties from type '{ kind: "Ctor9"; f0: _uniq_0; f1: _uniq_0; f2: _uniq_2; }': f0, f1, f2

[7m38[0m             _uniq_10(_uniq_16, _uniq_19),
[7m  [0m [91m                     ~~~~~~~~[0m
[96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m76[0m:[93m15[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor3" | "Ctor4"'. Did you mean '"Ctor3"'?

[7m76[0m         f0: { kind: "Ctor7" },
[7m  [0m [91m              ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m3[0m:[93m18[0m
    [7m3[0m type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4" };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_0'
[96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m77[0m:[93m15[0m - [91merror[0m[90m TS2820: [0mType '"Ctor3"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m77[0m         f1: { kind: "Ctor3" },
[7m  [0m [91m              ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m88[0m:[93m15[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor3" | "Ctor4"'. Did you mean '"Ctor3"'?

[7m88[0m         f0: { kind: "Ctor7" },
[7m  [0m [91m              ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m3[0m:[93m18[0m
    [7m3[0m type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4" };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_0'
[96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m89[0m:[93m15[0m - [91merror[0m[90m TS2820: [0mType '"Ctor3"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m89[0m         f1: { kind: "Ctor3" },
[7m  [0m [91m              ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m100[0m:[93m15[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor3" | "Ctor4"'. Did you mean '"Ctor3"'?

[7m100[0m         f0: { kind: "Ctor7" },
[7m   [0m [91m              ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m3[0m:[93m18[0m
    [7m3[0m type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4" };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_0'
[96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m101[0m:[93m15[0m - [91merror[0m[90m TS2820: [0mType '"Ctor3"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m101[0m         f1: { kind: "Ctor3" },
[7m   [0m [91m              ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m112[0m:[93m15[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor3" | "Ctor4"'. Did you mean '"Ctor3"'?

[7m112[0m         f0: { kind: "Ctor7" },
[7m   [0m [91m              ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m3[0m:[93m18[0m
    [7m3[0m type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4" };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_0'
[96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m113[0m:[93m15[0m - [91merror[0m[90m TS2820: [0mType '"Ctor3"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m113[0m         f1: { kind: "Ctor3" },
[7m   [0m [91m              ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m128[0m:[93m11[0m - [91merror[0m[90m TS2820: [0mType '"Ctor8"' is not assignable to type '"Ctor3" | "Ctor4"'. Did you mean '"Ctor3"'?

[7m128[0m           kind: "Ctor8",
[7m   [0m [91m          ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m3[0m:[93m18[0m
    [7m3[0m type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4" };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_0'
[96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m133[0m:[93m15[0m - [91merror[0m[90m TS2820: [0mType '"Ctor3"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m133[0m         f1: { kind: "Ctor3" },
[7m   [0m [91m              ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m145[0m:[93m11[0m - [91merror[0m[90m TS2820: [0mType '"Ctor8"' is not assignable to type '"Ctor3" | "Ctor4"'. Did you mean '"Ctor3"'?

[7m145[0m           kind: "Ctor8",
[7m   [0m [91m          ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m3[0m:[93m18[0m
    [7m3[0m type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4" };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_0'
[96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m150[0m:[93m15[0m - [91merror[0m[90m TS2820: [0mType '"Ctor3"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m150[0m         f1: { kind: "Ctor3" },
[7m   [0m [91m              ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m162[0m:[93m11[0m - [91merror[0m[90m TS2820: [0mType '"Ctor8"' is not assignable to type '"Ctor3" | "Ctor4"'. Did you mean '"Ctor3"'?

[7m162[0m           kind: "Ctor8",
[7m   [0m [91m          ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m3[0m:[93m18[0m
    [7m3[0m type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4" };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_0'
[96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m167[0m:[93m15[0m - [91merror[0m[90m TS2820: [0mType '"Ctor3"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m167[0m         f1: { kind: "Ctor3" },
[7m   [0m [91m              ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m179[0m:[93m11[0m - [91merror[0m[90m TS2820: [0mType '"Ctor8"' is not assignable to type '"Ctor3" | "Ctor4"'. Did you mean '"Ctor3"'?

[7m179[0m           kind: "Ctor8",
[7m   [0m [91m          ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m3[0m:[93m18[0m
    [7m3[0m type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4" };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_0'
[96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m184[0m:[93m15[0m - [91merror[0m[90m TS2820: [0mType '"Ctor3"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m184[0m         f1: { kind: "Ctor3" },
[7m   [0m [91m              ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m198[0m:[93m15[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor3" | "Ctor4"'. Did you mean '"Ctor3"'?

[7m198[0m         f0: { kind: "Ctor7" },
[7m   [0m [91m              ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m3[0m:[93m18[0m
    [7m3[0m type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4" };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_0'
[96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m199[0m:[93m15[0m - [91merror[0m[90m TS2820: [0mType '"Ctor3"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m199[0m         f1: { kind: "Ctor3" },
[7m   [0m [91m              ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m202[0m:[93m17[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor3" | "Ctor4"'. Did you mean '"Ctor3"'?

[7m202[0m           f0: { kind: "Ctor7" },
[7m   [0m [91m                ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m3[0m:[93m18[0m
    [7m3[0m type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4" };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_0'
[96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m203[0m:[93m17[0m - [91merror[0m[90m TS2820: [0mType '"Ctor3"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m203[0m           f1: { kind: "Ctor3" },
[7m   [0m [91m                ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m215[0m:[93m15[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor3" | "Ctor4"'. Did you mean '"Ctor3"'?

[7m215[0m         f0: { kind: "Ctor7" },
[7m   [0m [91m              ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m3[0m:[93m18[0m
    [7m3[0m type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4" };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_0'
[96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m216[0m:[93m15[0m - [91merror[0m[90m TS2820: [0mType '"Ctor3"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m216[0m         f1: { kind: "Ctor3" },
[7m   [0m [91m              ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m219[0m:[93m17[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor3" | "Ctor4"'. Did you mean '"Ctor3"'?

[7m219[0m           f0: { kind: "Ctor7" },
[7m   [0m [91m                ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m3[0m:[93m18[0m
    [7m3[0m type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4" };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_0'
[96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m220[0m:[93m17[0m - [91merror[0m[90m TS2820: [0mType '"Ctor3"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m220[0m           f1: { kind: "Ctor3" },
[7m   [0m [91m                ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m232[0m:[93m15[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor3" | "Ctor4"'. Did you mean '"Ctor3"'?

[7m232[0m         f0: { kind: "Ctor7" },
[7m   [0m [91m              ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m3[0m:[93m18[0m
    [7m3[0m type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4" };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_0'
[96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m233[0m:[93m15[0m - [91merror[0m[90m TS2820: [0mType '"Ctor3"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m233[0m         f1: { kind: "Ctor3" },
[7m   [0m [91m              ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m236[0m:[93m17[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor3" | "Ctor4"'. Did you mean '"Ctor3"'?

[7m236[0m           f0: { kind: "Ctor7" },
[7m   [0m [91m                ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m3[0m:[93m18[0m
    [7m3[0m type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4" };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_0'
[96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m237[0m:[93m17[0m - [91merror[0m[90m TS2820: [0mType '"Ctor3"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m237[0m           f1: { kind: "Ctor3" },
[7m   [0m [91m                ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m249[0m:[93m15[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor3" | "Ctor4"'. Did you mean '"Ctor3"'?

[7m249[0m         f0: { kind: "Ctor7" },
[7m   [0m [91m              ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m3[0m:[93m18[0m
    [7m3[0m type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4" };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_0'
[96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m250[0m:[93m15[0m - [91merror[0m[90m TS2820: [0mType '"Ctor3"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m250[0m         f1: { kind: "Ctor3" },
[7m   [0m [91m              ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m253[0m:[93m17[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor3" | "Ctor4"'. Did you mean '"Ctor3"'?

[7m253[0m           f0: { kind: "Ctor7" },
[7m   [0m [91m                ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m3[0m:[93m18[0m
    [7m3[0m type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4" };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_0'
[96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m254[0m:[93m17[0m - [91merror[0m[90m TS2820: [0mType '"Ctor3"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m254[0m           f1: { kind: "Ctor3" },
[7m   [0m [91m                ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m270[0m:[93m11[0m - [91merror[0m[90m TS2820: [0mType '"Ctor8"' is not assignable to type '"Ctor3" | "Ctor4"'. Did you mean '"Ctor3"'?

[7m270[0m           kind: "Ctor8",
[7m   [0m [91m          ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m3[0m:[93m18[0m
    [7m3[0m type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4" };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_0'
[96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m275[0m:[93m15[0m - [91merror[0m[90m TS2820: [0mType '"Ctor3"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m275[0m         f1: { kind: "Ctor3" },
[7m   [0m [91m              ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m278[0m:[93m17[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor3" | "Ctor4"'. Did you mean '"Ctor3"'?

[7m278[0m           f0: { kind: "Ctor7" },
[7m   [0m [91m                ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m3[0m:[93m18[0m
    [7m3[0m type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4" };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_0'
[96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m279[0m:[93m17[0m - [91merror[0m[90m TS2820: [0mType '"Ctor3"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m279[0m           f1: { kind: "Ctor3" },
[7m   [0m [91m                ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m292[0m:[93m11[0m - [91merror[0m[90m TS2820: [0mType '"Ctor8"' is not assignable to type '"Ctor3" | "Ctor4"'. Did you mean '"Ctor3"'?

[7m292[0m           kind: "Ctor8",
[7m   [0m [91m          ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m3[0m:[93m18[0m
    [7m3[0m type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4" };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_0'
[96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m297[0m:[93m15[0m - [91merror[0m[90m TS2820: [0mType '"Ctor3"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m297[0m         f1: { kind: "Ctor3" },
[7m   [0m [91m              ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m300[0m:[93m17[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor3" | "Ctor4"'. Did you mean '"Ctor3"'?

[7m300[0m           f0: { kind: "Ctor7" },
[7m   [0m [91m                ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m3[0m:[93m18[0m
    [7m3[0m type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4" };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_0'
[96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m301[0m:[93m17[0m - [91merror[0m[90m TS2820: [0mType '"Ctor3"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m301[0m           f1: { kind: "Ctor3" },
[7m   [0m [91m                ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m314[0m:[93m11[0m - [91merror[0m[90m TS2820: [0mType '"Ctor8"' is not assignable to type '"Ctor3" | "Ctor4"'. Did you mean '"Ctor3"'?

[7m314[0m           kind: "Ctor8",
[7m   [0m [91m          ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m3[0m:[93m18[0m
    [7m3[0m type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4" };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_0'
[96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m319[0m:[93m15[0m - [91merror[0m[90m TS2820: [0mType '"Ctor3"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m319[0m         f1: { kind: "Ctor3" },
[7m   [0m [91m              ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m322[0m:[93m17[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor3" | "Ctor4"'. Did you mean '"Ctor3"'?

[7m322[0m           f0: { kind: "Ctor7" },
[7m   [0m [91m                ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m3[0m:[93m18[0m
    [7m3[0m type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4" };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_0'
[96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m323[0m:[93m17[0m - [91merror[0m[90m TS2820: [0mType '"Ctor3"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m323[0m           f1: { kind: "Ctor3" },
[7m   [0m [91m                ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m336[0m:[93m11[0m - [91merror[0m[90m TS2820: [0mType '"Ctor8"' is not assignable to type '"Ctor3" | "Ctor4"'. Did you mean '"Ctor3"'?

[7m336[0m           kind: "Ctor8",
[7m   [0m [91m          ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m3[0m:[93m18[0m
    [7m3[0m type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4" };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_0'
[96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m341[0m:[93m15[0m - [91merror[0m[90m TS2820: [0mType '"Ctor3"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m341[0m         f1: { kind: "Ctor3" },
[7m   [0m [91m              ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m344[0m:[93m17[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor3" | "Ctor4"'. Did you mean '"Ctor3"'?

[7m344[0m           f0: { kind: "Ctor7" },
[7m   [0m [91m                ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m3[0m:[93m18[0m
    [7m3[0m type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4" };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_0'
[96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m345[0m:[93m17[0m - [91merror[0m[90m TS2820: [0mType '"Ctor3"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m345[0m           f1: { kind: "Ctor3" },
[7m   [0m [91m                ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m361[0m:[93m11[0m - [91merror[0m[90m TS2820: [0mType '"Ctor8"' is not assignable to type '"Ctor3" | "Ctor4"'. Did you mean '"Ctor3"'?

[7m361[0m           kind: "Ctor8",
[7m   [0m [91m          ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m3[0m:[93m18[0m
    [7m3[0m type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4" };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_0'
[96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m376[0m:[93m15[0m - [91merror[0m[90m TS2820: [0mType '"Ctor3"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m376[0m         f1: { kind: "Ctor3" },
[7m   [0m [91m              ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m388[0m:[93m11[0m - [91merror[0m[90m TS2820: [0mType '"Ctor8"' is not assignable to type '"Ctor3" | "Ctor4"'. Did you mean '"Ctor3"'?

[7m388[0m           kind: "Ctor8",
[7m   [0m [91m          ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m3[0m:[93m18[0m
    [7m3[0m type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4" };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_0'
[96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m403[0m:[93m15[0m - [91merror[0m[90m TS2820: [0mType '"Ctor3"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m403[0m         f1: { kind: "Ctor3" },
[7m   [0m [91m              ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m415[0m:[93m11[0m - [91merror[0m[90m TS2820: [0mType '"Ctor8"' is not assignable to type '"Ctor3" | "Ctor4"'. Did you mean '"Ctor3"'?

[7m415[0m           kind: "Ctor8",
[7m   [0m [91m          ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m3[0m:[93m18[0m
    [7m3[0m type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4" };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_0'
[96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m430[0m:[93m15[0m - [91merror[0m[90m TS2820: [0mType '"Ctor3"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m430[0m         f1: { kind: "Ctor3" },
[7m   [0m [91m              ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m442[0m:[93m11[0m - [91merror[0m[90m TS2820: [0mType '"Ctor8"' is not assignable to type '"Ctor3" | "Ctor4"'. Did you mean '"Ctor3"'?

[7m442[0m           kind: "Ctor8",
[7m   [0m [91m          ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m3[0m:[93m18[0m
    [7m3[0m type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4" };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_0'
[96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m457[0m:[93m15[0m - [91merror[0m[90m TS2820: [0mType '"Ctor3"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m457[0m         f1: { kind: "Ctor3" },
[7m   [0m [91m              ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_8.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_1 };
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
    2366, 2345, 2820, 2820, 2820, 2820, 2820,
    2820, 2820, 2820, 2820, 2820, 2820, 2820,
    2820, 2820, 2820, 2820, 2820, 2820, 2820,
    2820, 2820, 2820, 2820, 2820, 2820, 2820,
    2820, 2820, 2820, 2820, 2820, 2820, 2820,
    2820, 2820, 2820, 2820, 2820, 2820, 2820,
    2820, 2820, 2820, 2820, 2820, 2820, 2820,
    2820, 2820, 2820, 2820, 2820, 2820, 2820,
    2820, 2820
  ]
}
