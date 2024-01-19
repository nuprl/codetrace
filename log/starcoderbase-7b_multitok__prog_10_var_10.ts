/home/franlucc/.npm_packages/_npx/1bf7c3c15bf47d04/node_modules/ts-node/src/index.ts:859
    return new TSError(diagnosticText, diagnosticCodes, diagnostics);
           ^
TSError: ⨯ Unable to compile TypeScript:
[96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m10[0m:[93m49[0m - [91merror[0m[90m TS7008: [0mMember 'f' implicitly has an 'any' type.

[7m10[0m   | { kind: "Ctor10"; f0: _uniq_3; f1: _uniq_2; f };
[7m  [0m [91m                                                ~[0m
[96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m41[0m:[93m22[0m - [91merror[0m[90m TS2345: [0mArgument of type '_uniq_1' is not assignable to parameter of type '_uniq_2'.
  Type '{ kind: "Ctor5"; }' is not assignable to type '_uniq_2'.
    Type '{ kind: "Ctor5"; }' is missing the following properties from type '{ kind: "Ctor8"; f0: _uniq_2; f1: _uniq_0; f2: _uniq_1; }': f0, f1, f2

[7m41[0m             _uniq_10(_uniq_18, _uniq_19)
[7m  [0m [91m                     ~~~~~~~~[0m
[96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m80[0m:[93m15[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m80[0m         f2: { kind: "Ctor7" },
[7m  [0m [91m              ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m92[0m:[93m15[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m92[0m         f2: { kind: "Ctor7" },
[7m  [0m [91m              ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m104[0m:[93m15[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m104[0m         f2: { kind: "Ctor7" },
[7m   [0m [91m              ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m116[0m:[93m15[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m116[0m         f2: { kind: "Ctor7" },
[7m   [0m [91m              ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m133[0m:[93m17[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m133[0m           f2: { kind: "Ctor7" },
[7m   [0m [91m                ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m136[0m:[93m15[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m136[0m         f2: { kind: "Ctor7" },
[7m   [0m [91m              ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m150[0m:[93m17[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m150[0m           f2: { kind: "Ctor7" },
[7m   [0m [91m                ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m153[0m:[93m15[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m153[0m         f2: { kind: "Ctor7" },
[7m   [0m [91m              ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m167[0m:[93m17[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m167[0m           f2: { kind: "Ctor7" },
[7m   [0m [91m                ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m170[0m:[93m15[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m170[0m         f2: { kind: "Ctor7" },
[7m   [0m [91m              ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m184[0m:[93m17[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m184[0m           f2: { kind: "Ctor7" },
[7m   [0m [91m                ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m187[0m:[93m15[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m187[0m         f2: { kind: "Ctor7" },
[7m   [0m [91m              ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m203[0m:[93m11[0m - [91merror[0m[90m TS2820: [0mType '"Ctor8"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m203[0m           kind: "Ctor8",
[7m   [0m [91m          ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m204[0m:[93m17[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m204[0m           f0: { kind: "Ctor7" },
[7m   [0m [91m                ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m220[0m:[93m11[0m - [91merror[0m[90m TS2820: [0mType '"Ctor8"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m220[0m           kind: "Ctor8",
[7m   [0m [91m          ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m221[0m:[93m17[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m221[0m           f0: { kind: "Ctor7" },
[7m   [0m [91m                ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m237[0m:[93m11[0m - [91merror[0m[90m TS2820: [0mType '"Ctor8"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m237[0m           kind: "Ctor8",
[7m   [0m [91m          ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m238[0m:[93m17[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m238[0m           f0: { kind: "Ctor7" },
[7m   [0m [91m                ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m254[0m:[93m11[0m - [91merror[0m[90m TS2820: [0mType '"Ctor8"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m254[0m           kind: "Ctor8",
[7m   [0m [91m          ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m255[0m:[93m17[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m255[0m           f0: { kind: "Ctor7" },
[7m   [0m [91m                ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m275[0m:[93m17[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m275[0m           f2: { kind: "Ctor7" },
[7m   [0m [91m                ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m279[0m:[93m11[0m - [91merror[0m[90m TS2820: [0mType '"Ctor8"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m279[0m           kind: "Ctor8",
[7m   [0m [91m          ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m280[0m:[93m17[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m280[0m           f0: { kind: "Ctor7" },
[7m   [0m [91m                ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m297[0m:[93m17[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m297[0m           f2: { kind: "Ctor7" },
[7m   [0m [91m                ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m301[0m:[93m11[0m - [91merror[0m[90m TS2820: [0mType '"Ctor8"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m301[0m           kind: "Ctor8",
[7m   [0m [91m          ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m302[0m:[93m17[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m302[0m           f0: { kind: "Ctor7" },
[7m   [0m [91m                ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m319[0m:[93m17[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m319[0m           f2: { kind: "Ctor7" },
[7m   [0m [91m                ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m323[0m:[93m11[0m - [91merror[0m[90m TS2820: [0mType '"Ctor8"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m323[0m           kind: "Ctor8",
[7m   [0m [91m          ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m324[0m:[93m17[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m324[0m           f0: { kind: "Ctor7" },
[7m   [0m [91m                ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m341[0m:[93m17[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m341[0m           f2: { kind: "Ctor7" },
[7m   [0m [91m                ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m345[0m:[93m11[0m - [91merror[0m[90m TS2820: [0mType '"Ctor8"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m345[0m           kind: "Ctor8",
[7m   [0m [91m          ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m346[0m:[93m17[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m346[0m           f0: { kind: "Ctor7" },
[7m   [0m [91m                ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m368[0m:[93m19[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m368[0m             f2: { kind: "Ctor7" },
[7m   [0m [91m                  ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m372[0m:[93m13[0m - [91merror[0m[90m TS2820: [0mType '"Ctor8"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m372[0m             kind: "Ctor8",
[7m   [0m [91m            ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m373[0m:[93m19[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m373[0m             f0: { kind: "Ctor7" },
[7m   [0m [91m                  ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m379[0m:[93m15[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m379[0m         f2: { kind: "Ctor7" },
[7m   [0m [91m              ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m395[0m:[93m19[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m395[0m             f2: { kind: "Ctor7" },
[7m   [0m [91m                  ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m399[0m:[93m13[0m - [91merror[0m[90m TS2820: [0mType '"Ctor8"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m399[0m             kind: "Ctor8",
[7m   [0m [91m            ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m400[0m:[93m19[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m400[0m             f0: { kind: "Ctor7" },
[7m   [0m [91m                  ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m406[0m:[93m15[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m406[0m         f2: { kind: "Ctor7" },
[7m   [0m [91m              ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m422[0m:[93m19[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m422[0m             f2: { kind: "Ctor7" },
[7m   [0m [91m                  ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m426[0m:[93m13[0m - [91merror[0m[90m TS2820: [0mType '"Ctor8"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m426[0m             kind: "Ctor8",
[7m   [0m [91m            ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m427[0m:[93m19[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m427[0m             f0: { kind: "Ctor7" },
[7m   [0m [91m                  ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m433[0m:[93m15[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m433[0m         f2: { kind: "Ctor7" },
[7m   [0m [91m              ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m449[0m:[93m19[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m449[0m             f2: { kind: "Ctor7" },
[7m   [0m [91m                  ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m453[0m:[93m13[0m - [91merror[0m[90m TS2820: [0mType '"Ctor8"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m453[0m             kind: "Ctor8",
[7m   [0m [91m            ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m454[0m:[93m19[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m454[0m             f0: { kind: "Ctor7" },
[7m   [0m [91m                  ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m4[0m:[93m18[0m
    [7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6"; f0: _uniq_1 };
    [7m [0m [96m                 ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m460[0m:[93m15[0m - [91merror[0m[90m TS2820: [0mType '"Ctor7"' is not assignable to type '"Ctor5" | "Ctor6"'. Did you mean '"Ctor5"'?

[7m460[0m         f2: { kind: "Ctor7" },
[7m   [0m [91m              ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_10/var_10.ts[0m:[93m4[0m:[93m18[0m
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
    7008, 2345, 2820, 2820, 2820, 2820,
    2820, 2820, 2820, 2820, 2820, 2820,
    2820, 2820, 2820, 2820, 2820, 2820,
    2820, 2820, 2820, 2820, 2820, 2820,
    2820, 2820, 2820, 2820, 2820, 2820,
    2820, 2820, 2820, 2820, 2820, 2820,
    2820, 2820, 2820, 2820, 2820, 2820,
    2820, 2820, 2820, 2820, 2820, 2820,
    2820, 2820
  ]
}
