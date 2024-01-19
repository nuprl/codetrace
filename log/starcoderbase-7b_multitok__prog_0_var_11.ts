/home/franlucc/.npm_packages/_npx/1bf7c3c15bf47d04/node_modules/ts-node/src/index.ts:859
    return new TSError(diagnosticText, diagnosticCodes, diagnostics);
           ^
TSError: ⨯ Unable to compile TypeScript:
[96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m10[0m:[93m36[0m - [91merror[0m[90m TS7008: [0mMember 'f' implicitly has an 'any' type.

[7m10[0m   | { kind: "Ctor10"; f0: _uniq_3; f; f2: _uniq_2 };
[7m  [0m [91m                                   ~[0m
[96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m35[0m:[93m26[0m - [91merror[0m[90m TS2322: [0mType '_uniq_2' is not assignable to type '_uniq_0'.
  Type '{ kind: "Ctor7"; }' is not assignable to type '_uniq_0'.
    Property 'f0' is missing in type '{ kind: "Ctor7"; }' but required in type '{ kind: "Ctor4"; f0: _uniq_0; }'.

[7m35[0m         { kind: "Ctor6", f0: _uniq_17, f1: _uniq_10(_uniq_16) },
[7m  [0m [91m                         ~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m3[0m:[93m53[0m
    [7m3[0m type _uniq_0 = { kind: "Ctor3" } | { kind: "Ctor4"; f0: _uniq_0 };
    [7m [0m [96m                                                    ~~[0m
    'f0' is declared here.
[96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m50[0m:[93m15[0m - [91merror[0m[90m TS2820: [0mType '"Ctor4"' is not assignable to type '"Ctor7" | "Ctor8"'. Did you mean '"Ctor7"'?

[7m50[0m         f1: { kind: "Ctor4", f0: { kind: "Ctor3" } },
[7m  [0m [91m              ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m6[0m:[93m7[0m
    [7m6[0m   | { kind: "Ctor7" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_2'
[96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m50[0m:[93m36[0m - [91merror[0m[90m TS2820: [0mType '"Ctor3"' is not assignable to type '"Ctor7" | "Ctor8"'. Did you mean '"Ctor7"'?

[7m50[0m         f1: { kind: "Ctor4", f0: { kind: "Ctor3" } },
[7m  [0m [91m                                   ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m6[0m:[93m7[0m
    [7m6[0m   | { kind: "Ctor7" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_2'
[96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m53[0m:[93m13[0m - [91merror[0m[90m TS2820: [0mType '"Ctor4"' is not assignable to type '"Ctor7" | "Ctor8"'. Did you mean '"Ctor7"'?

[7m53[0m       f1: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
[7m  [0m [91m            ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m6[0m:[93m7[0m
    [7m6[0m   | { kind: "Ctor7" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_2'
[96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m53[0m:[93m34[0m - [91merror[0m[90m TS2820: [0mType '"Ctor4"' is not assignable to type '"Ctor7" | "Ctor8"'. Did you mean '"Ctor7"'?

[7m53[0m       f1: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
[7m  [0m [91m                                 ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m6[0m:[93m7[0m
    [7m6[0m   | { kind: "Ctor7" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_2'
[96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m53[0m:[93m55[0m - [91merror[0m[90m TS2820: [0mType '"Ctor3"' is not assignable to type '"Ctor7" | "Ctor8"'. Did you mean '"Ctor7"'?

[7m53[0m       f1: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
[7m  [0m [91m                                                      ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m6[0m:[93m7[0m
    [7m6[0m   | { kind: "Ctor7" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_2'
[96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m70[0m:[93m13[0m - [91merror[0m[90m TS2820: [0mType '"Ctor4"' is not assignable to type '"Ctor7" | "Ctor8"'. Did you mean '"Ctor7"'?

[7m70[0m       f1: { kind: "Ctor4", f0: { kind: "Ctor3" } },
[7m  [0m [91m            ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m6[0m:[93m7[0m
    [7m6[0m   | { kind: "Ctor7" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_2'
[96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m70[0m:[93m34[0m - [91merror[0m[90m TS2820: [0mType '"Ctor3"' is not assignable to type '"Ctor7" | "Ctor8"'. Did you mean '"Ctor7"'?

[7m70[0m       f1: { kind: "Ctor4", f0: { kind: "Ctor3" } },
[7m  [0m [91m                                 ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m6[0m:[93m7[0m
    [7m6[0m   | { kind: "Ctor7" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_2'
[96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m74[0m:[93m15[0m - [91merror[0m[90m TS2820: [0mType '"Ctor4"' is not assignable to type '"Ctor7" | "Ctor8"'. Did you mean '"Ctor7"'?

[7m74[0m         f1: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
[7m  [0m [91m              ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m6[0m:[93m7[0m
    [7m6[0m   | { kind: "Ctor7" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_2'
[96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m74[0m:[93m36[0m - [91merror[0m[90m TS2820: [0mType '"Ctor4"' is not assignable to type '"Ctor7" | "Ctor8"'. Did you mean '"Ctor7"'?

[7m74[0m         f1: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
[7m  [0m [91m                                   ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m6[0m:[93m7[0m
    [7m6[0m   | { kind: "Ctor7" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_2'
[96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m74[0m:[93m57[0m - [91merror[0m[90m TS2820: [0mType '"Ctor3"' is not assignable to type '"Ctor7" | "Ctor8"'. Did you mean '"Ctor7"'?

[7m74[0m         f1: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
[7m  [0m [91m                                                        ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m6[0m:[93m7[0m
    [7m6[0m   | { kind: "Ctor7" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_2'
[96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m100[0m:[93m17[0m - [91merror[0m[90m TS2820: [0mType '"Ctor3"' is not assignable to type '"Ctor7" | "Ctor8"'. Did you mean '"Ctor7"'?

[7m100[0m           f1: { kind: "Ctor3" },
[7m   [0m [91m                ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m6[0m:[93m7[0m
    [7m6[0m   | { kind: "Ctor7" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_2'
[96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m103[0m:[93m15[0m - [91merror[0m[90m TS2820: [0mType '"Ctor4"' is not assignable to type '"Ctor7" | "Ctor8"'. Did you mean '"Ctor7"'?

[7m103[0m         f1: { kind: "Ctor4", f0: { kind: "Ctor3" } },
[7m   [0m [91m              ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m6[0m:[93m7[0m
    [7m6[0m   | { kind: "Ctor7" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_2'
[96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m103[0m:[93m36[0m - [91merror[0m[90m TS2820: [0mType '"Ctor3"' is not assignable to type '"Ctor7" | "Ctor8"'. Did you mean '"Ctor7"'?

[7m103[0m         f1: { kind: "Ctor4", f0: { kind: "Ctor3" } },
[7m   [0m [91m                                   ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m6[0m:[93m7[0m
    [7m6[0m   | { kind: "Ctor7" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_2'
[96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m107[0m:[93m17[0m - [91merror[0m[90m TS2820: [0mType '"Ctor4"' is not assignable to type '"Ctor7" | "Ctor8"'. Did you mean '"Ctor7"'?

[7m107[0m           f1: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
[7m   [0m [91m                ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m6[0m:[93m7[0m
    [7m6[0m   | { kind: "Ctor7" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_2'
[96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m107[0m:[93m38[0m - [91merror[0m[90m TS2820: [0mType '"Ctor4"' is not assignable to type '"Ctor7" | "Ctor8"'. Did you mean '"Ctor7"'?

[7m107[0m           f1: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
[7m   [0m [91m                                     ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m6[0m:[93m7[0m
    [7m6[0m   | { kind: "Ctor7" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_2'
[96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m107[0m:[93m59[0m - [91merror[0m[90m TS2820: [0mType '"Ctor3"' is not assignable to type '"Ctor7" | "Ctor8"'. Did you mean '"Ctor7"'?

[7m107[0m           f1: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
[7m   [0m [91m                                                          ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m6[0m:[93m7[0m
    [7m6[0m   | { kind: "Ctor7" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_2'
[96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m112[0m:[93m9[0m - [91merror[0m[90m TS2820: [0mType '"Ctor4"' is not assignable to type '"Ctor7" | "Ctor8"'. Did you mean '"Ctor7"'?

[7m112[0m         kind: "Ctor4",
[7m   [0m [91m        ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m6[0m:[93m7[0m
    [7m6[0m   | { kind: "Ctor7" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_2'
[96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m113[0m:[93m15[0m - [91merror[0m[90m TS2820: [0mType '"Ctor4"' is not assignable to type '"Ctor7" | "Ctor8"'. Did you mean '"Ctor7"'?

[7m113[0m         f0: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
[7m   [0m [91m              ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m6[0m:[93m7[0m
    [7m6[0m   | { kind: "Ctor7" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_2'
[96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m113[0m:[93m36[0m - [91merror[0m[90m TS2820: [0mType '"Ctor4"' is not assignable to type '"Ctor7" | "Ctor8"'. Did you mean '"Ctor7"'?

[7m113[0m         f0: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
[7m   [0m [91m                                   ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m6[0m:[93m7[0m
    [7m6[0m   | { kind: "Ctor7" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_2'
[96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m113[0m:[93m57[0m - [91merror[0m[90m TS2820: [0mType '"Ctor3"' is not assignable to type '"Ctor7" | "Ctor8"'. Did you mean '"Ctor7"'?

[7m113[0m         f0: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
[7m   [0m [91m                                                        ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m6[0m:[93m7[0m
    [7m6[0m   | { kind: "Ctor7" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_2'
[96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m121[0m:[93m13[0m - [91merror[0m[90m TS2820: [0mType '"Ctor4"' is not assignable to type '"Ctor7" | "Ctor8"'. Did you mean '"Ctor7"'?

[7m121[0m             kind: "Ctor4",
[7m   [0m [91m            ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m6[0m:[93m7[0m
    [7m6[0m   | { kind: "Ctor7" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_2'
[96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m123[0m:[93m15[0m - [91merror[0m[90m TS2820: [0mType '"Ctor4"' is not assignable to type '"Ctor7" | "Ctor8"'. Did you mean '"Ctor7"'?

[7m123[0m               kind: "Ctor4",
[7m   [0m [91m              ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m6[0m:[93m7[0m
    [7m6[0m   | { kind: "Ctor7" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_2'
[96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m125[0m:[93m17[0m - [91merror[0m[90m TS2820: [0mType '"Ctor4"' is not assignable to type '"Ctor7" | "Ctor8"'. Did you mean '"Ctor7"'?

[7m125[0m                 kind: "Ctor4",
[7m   [0m [91m                ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m6[0m:[93m7[0m
    [7m6[0m   | { kind: "Ctor7" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_2'
[96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m126[0m:[93m23[0m - [91merror[0m[90m TS2820: [0mType '"Ctor4"' is not assignable to type '"Ctor7" | "Ctor8"'. Did you mean '"Ctor7"'?

[7m126[0m                 f0: { kind: "Ctor4", f0: { kind: "Ctor3" } },
[7m   [0m [91m                      ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m6[0m:[93m7[0m
    [7m6[0m   | { kind: "Ctor7" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_2'
[96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m126[0m:[93m44[0m - [91merror[0m[90m TS2820: [0mType '"Ctor3"' is not assignable to type '"Ctor7" | "Ctor8"'. Did you mean '"Ctor7"'?

[7m126[0m                 f0: { kind: "Ctor4", f0: { kind: "Ctor3" } },
[7m   [0m [91m                                           ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m6[0m:[93m7[0m
    [7m6[0m   | { kind: "Ctor7" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_2'
[96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m133[0m:[93m11[0m - [91merror[0m[90m TS2820: [0mType '"Ctor4"' is not assignable to type '"Ctor7" | "Ctor8"'. Did you mean '"Ctor7"'?

[7m133[0m           kind: "Ctor4",
[7m   [0m [91m          ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m6[0m:[93m7[0m
    [7m6[0m   | { kind: "Ctor7" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_2'
[96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m135[0m:[93m13[0m - [91merror[0m[90m TS2820: [0mType '"Ctor4"' is not assignable to type '"Ctor7" | "Ctor8"'. Did you mean '"Ctor7"'?

[7m135[0m             kind: "Ctor4",
[7m   [0m [91m            ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m6[0m:[93m7[0m
    [7m6[0m   | { kind: "Ctor7" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_2'
[96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m137[0m:[93m15[0m - [91merror[0m[90m TS2820: [0mType '"Ctor4"' is not assignable to type '"Ctor7" | "Ctor8"'. Did you mean '"Ctor7"'?

[7m137[0m               kind: "Ctor4",
[7m   [0m [91m              ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m6[0m:[93m7[0m
    [7m6[0m   | { kind: "Ctor7" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_2'
[96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m139[0m:[93m17[0m - [91merror[0m[90m TS2820: [0mType '"Ctor4"' is not assignable to type '"Ctor7" | "Ctor8"'. Did you mean '"Ctor7"'?

[7m139[0m                 kind: "Ctor4",
[7m   [0m [91m                ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m6[0m:[93m7[0m
    [7m6[0m   | { kind: "Ctor7" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_2'
[96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m140[0m:[93m23[0m - [91merror[0m[90m TS2820: [0mType '"Ctor4"' is not assignable to type '"Ctor7" | "Ctor8"'. Did you mean '"Ctor7"'?

[7m140[0m                 f0: { kind: "Ctor4", f0: { kind: "Ctor3" } },
[7m   [0m [91m                      ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m6[0m:[93m7[0m
    [7m6[0m   | { kind: "Ctor7" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_2'
[96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m140[0m:[93m44[0m - [91merror[0m[90m TS2820: [0mType '"Ctor3"' is not assignable to type '"Ctor7" | "Ctor8"'. Did you mean '"Ctor7"'?

[7m140[0m                 f0: { kind: "Ctor4", f0: { kind: "Ctor3" } },
[7m   [0m [91m                                           ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m6[0m:[93m7[0m
    [7m6[0m   | { kind: "Ctor7" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_2'
[96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m149[0m:[93m13[0m - [91merror[0m[90m TS2820: [0mType '"Ctor4"' is not assignable to type '"Ctor7" | "Ctor8"'. Did you mean '"Ctor7"'?

[7m149[0m             kind: "Ctor4",
[7m   [0m [91m            ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m6[0m:[93m7[0m
    [7m6[0m   | { kind: "Ctor7" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_2'
[96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m151[0m:[93m15[0m - [91merror[0m[90m TS2820: [0mType '"Ctor4"' is not assignable to type '"Ctor7" | "Ctor8"'. Did you mean '"Ctor7"'?

[7m151[0m               kind: "Ctor4",
[7m   [0m [91m              ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m6[0m:[93m7[0m
    [7m6[0m   | { kind: "Ctor7" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_2'
[96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m153[0m:[93m17[0m - [91merror[0m[90m TS2820: [0mType '"Ctor4"' is not assignable to type '"Ctor7" | "Ctor8"'. Did you mean '"Ctor7"'?

[7m153[0m                 kind: "Ctor4",
[7m   [0m [91m                ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m6[0m:[93m7[0m
    [7m6[0m   | { kind: "Ctor7" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_2'
[96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m155[0m:[93m19[0m - [91merror[0m[90m TS2820: [0mType '"Ctor4"' is not assignable to type '"Ctor7" | "Ctor8"'. Did you mean '"Ctor7"'?

[7m155[0m                   kind: "Ctor4",
[7m   [0m [91m                  ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m6[0m:[93m7[0m
    [7m6[0m   | { kind: "Ctor7" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_2'
[96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m157[0m:[93m21[0m - [91merror[0m[90m TS2820: [0mType '"Ctor4"' is not assignable to type '"Ctor7" | "Ctor8"'. Did you mean '"Ctor7"'?

[7m157[0m                     kind: "Ctor4",
[7m   [0m [91m                    ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m6[0m:[93m7[0m
    [7m6[0m   | { kind: "Ctor7" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_2'
[96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m158[0m:[93m27[0m - [91merror[0m[90m TS2820: [0mType '"Ctor4"' is not assignable to type '"Ctor7" | "Ctor8"'. Did you mean '"Ctor7"'?

[7m158[0m                     f0: { kind: "Ctor4", f0: { kind: "Ctor3" } },
[7m   [0m [91m                          ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m6[0m:[93m7[0m
    [7m6[0m   | { kind: "Ctor7" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_2'
[96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m158[0m:[93m48[0m - [91merror[0m[90m TS2820: [0mType '"Ctor3"' is not assignable to type '"Ctor7" | "Ctor8"'. Did you mean '"Ctor7"'?

[7m158[0m                     f0: { kind: "Ctor4", f0: { kind: "Ctor3" } },
[7m   [0m [91m                                               ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m6[0m:[93m7[0m
    [7m6[0m   | { kind: "Ctor7" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_2'
[96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m241[0m:[93m13[0m - [91merror[0m[90m TS2820: [0mType '"Ctor3"' is not assignable to type '"Ctor7" | "Ctor8"'. Did you mean '"Ctor7"'?

[7m241[0m       f1: { kind: "Ctor3" },
[7m   [0m [91m            ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m6[0m:[93m7[0m
    [7m6[0m   | { kind: "Ctor7" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_2'
[96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m245[0m:[93m15[0m - [91merror[0m[90m TS2820: [0mType '"Ctor4"' is not assignable to type '"Ctor7" | "Ctor8"'. Did you mean '"Ctor7"'?

[7m245[0m         f1: { kind: "Ctor4", f0: { kind: "Ctor3" } },
[7m   [0m [91m              ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m6[0m:[93m7[0m
    [7m6[0m   | { kind: "Ctor7" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_2'
[96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m245[0m:[93m36[0m - [91merror[0m[90m TS2820: [0mType '"Ctor3"' is not assignable to type '"Ctor7" | "Ctor8"'. Did you mean '"Ctor7"'?

[7m245[0m         f1: { kind: "Ctor4", f0: { kind: "Ctor3" } },
[7m   [0m [91m                                   ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m6[0m:[93m7[0m
    [7m6[0m   | { kind: "Ctor7" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_2'
[96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m249[0m:[93m17[0m - [91merror[0m[90m TS2820: [0mType '"Ctor4"' is not assignable to type '"Ctor7" | "Ctor8"'. Did you mean '"Ctor7"'?

[7m249[0m           f1: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
[7m   [0m [91m                ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m6[0m:[93m7[0m
    [7m6[0m   | { kind: "Ctor7" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_2'
[96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m249[0m:[93m38[0m - [91merror[0m[90m TS2820: [0mType '"Ctor4"' is not assignable to type '"Ctor7" | "Ctor8"'. Did you mean '"Ctor7"'?

[7m249[0m           f1: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
[7m   [0m [91m                                     ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m6[0m:[93m7[0m
    [7m6[0m   | { kind: "Ctor7" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_2'
[96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m249[0m:[93m59[0m - [91merror[0m[90m TS2820: [0mType '"Ctor3"' is not assignable to type '"Ctor7" | "Ctor8"'. Did you mean '"Ctor7"'?

[7m249[0m           f1: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
[7m   [0m [91m                                                          ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m6[0m:[93m7[0m
    [7m6[0m   | { kind: "Ctor7" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_2'
[96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m276[0m:[93m17[0m - [91merror[0m[90m TS2820: [0mType '"Ctor4"' is not assignable to type '"Ctor7" | "Ctor8"'. Did you mean '"Ctor7"'?

[7m276[0m           f1: { kind: "Ctor4", f0: { kind: "Ctor3" } },
[7m   [0m [91m                ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m6[0m:[93m7[0m
    [7m6[0m   | { kind: "Ctor7" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_2'
[96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m276[0m:[93m38[0m - [91merror[0m[90m TS2820: [0mType '"Ctor3"' is not assignable to type '"Ctor7" | "Ctor8"'. Did you mean '"Ctor7"'?

[7m276[0m           f1: { kind: "Ctor4", f0: { kind: "Ctor3" } },
[7m   [0m [91m                                     ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m6[0m:[93m7[0m
    [7m6[0m   | { kind: "Ctor7" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_2'
[96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m279[0m:[93m15[0m - [91merror[0m[90m TS2820: [0mType '"Ctor3"' is not assignable to type '"Ctor7" | "Ctor8"'. Did you mean '"Ctor7"'?

[7m279[0m         f1: { kind: "Ctor3" },
[7m   [0m [91m              ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m6[0m:[93m7[0m
    [7m6[0m   | { kind: "Ctor7" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_2'
[96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m282[0m:[93m13[0m - [91merror[0m[90m TS2820: [0mType '"Ctor3"' is not assignable to type '"Ctor7" | "Ctor8"'. Did you mean '"Ctor7"'?

[7m282[0m       f1: { kind: "Ctor3" },
[7m   [0m [91m            ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m6[0m:[93m7[0m
    [7m6[0m   | { kind: "Ctor7" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_2'
[96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m286[0m:[93m15[0m - [91merror[0m[90m TS2820: [0mType '"Ctor4"' is not assignable to type '"Ctor7" | "Ctor8"'. Did you mean '"Ctor7"'?

[7m286[0m         f1: { kind: "Ctor4", f0: { kind: "Ctor3" } },
[7m   [0m [91m              ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m6[0m:[93m7[0m
    [7m6[0m   | { kind: "Ctor7" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_2'
[96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m286[0m:[93m36[0m - [91merror[0m[90m TS2820: [0mType '"Ctor3"' is not assignable to type '"Ctor7" | "Ctor8"'. Did you mean '"Ctor7"'?

[7m286[0m         f1: { kind: "Ctor4", f0: { kind: "Ctor3" } },
[7m   [0m [91m                                   ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m6[0m:[93m7[0m
    [7m6[0m   | { kind: "Ctor7" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_2'
[96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m290[0m:[93m17[0m - [91merror[0m[90m TS2820: [0mType '"Ctor4"' is not assignable to type '"Ctor7" | "Ctor8"'. Did you mean '"Ctor7"'?

[7m290[0m           f1: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
[7m   [0m [91m                ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m6[0m:[93m7[0m
    [7m6[0m   | { kind: "Ctor7" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_2'
[96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m290[0m:[93m38[0m - [91merror[0m[90m TS2820: [0mType '"Ctor4"' is not assignable to type '"Ctor7" | "Ctor8"'. Did you mean '"Ctor7"'?

[7m290[0m           f1: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
[7m   [0m [91m                                     ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m6[0m:[93m7[0m
    [7m6[0m   | { kind: "Ctor7" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_2'
[96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m290[0m:[93m59[0m - [91merror[0m[90m TS2820: [0mType '"Ctor3"' is not assignable to type '"Ctor7" | "Ctor8"'. Did you mean '"Ctor7"'?

[7m290[0m           f1: { kind: "Ctor4", f0: { kind: "Ctor4", f0: { kind: "Ctor3" } } },
[7m   [0m [91m                                                          ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_0/var_11.ts[0m:[93m6[0m:[93m7[0m
    [7m6[0m   | { kind: "Ctor7" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_2'

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
    7008, 2322, 2820, 2820, 2820, 2820, 2820,
    2820, 2820, 2820, 2820, 2820, 2820, 2820,
    2820, 2820, 2820, 2820, 2820, 2820, 2820,
    2820, 2820, 2820, 2820, 2820, 2820, 2820,
    2820, 2820, 2820, 2820, 2820, 2820, 2820,
    2820, 2820, 2820, 2820, 2820, 2820, 2820,
    2820, 2820, 2820, 2820, 2820, 2820, 2820,
    2820, 2820, 2820, 2820, 2820, 2820
  ]
}
