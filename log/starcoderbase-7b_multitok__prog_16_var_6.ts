/home/franlucc/.npm_packages/_npx/1bf7c3c15bf47d04/node_modules/ts-node/src/index.ts:859
    return new TSError(diagnosticText, diagnosticCodes, diagnostics);
           ^
TSError: ⨯ Unable to compile TypeScript:
[96mstarcoderbase-7b/multitok/prog_16/var_6.ts[0m:[93m27[0m:[93m31[0m - [91merror[0m[90m TS2339: [0mProperty 'f2' does not exist on type '{ kind: "Ctor5"; f0: _uniq_1; f1: _uniq_1; } | { kind: "Ctor5"; f0: _uniq_1; f1: _uniq_1; f2: _uniq_1; }'.
  Property 'f2' does not exist on type '{ kind: "Ctor5"; f0: _uniq_1; f1: _uniq_1; }'.

[7m27[0m       let _uniq_14 = _uniq_11.f2;
[7m  [0m [91m                              ~~[0m
[96mstarcoderbase-7b/multitok/prog_16/var_6.ts[0m:[93m30[0m:[93m30[0m - [91merror[0m[90m TS2345: [0mArgument of type '_uniq_1' is not assignable to parameter of type '_uniq_0'.
  Type '{ kind: "Ctor4"; }' is not assignable to type '_uniq_0'.
    Property 'f0' is missing in type '{ kind: "Ctor4"; }' but required in type '{ kind: "Ctor3"; f0: _uniq_0; }'.

[7m30[0m       return _uniq_6(_uniq_6(_uniq_12, _uniq_7(_uniq_13)), _uniq_7(_uniq_14));
[7m  [0m [91m                             ~~~~~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_16/var_6.ts[0m:[93m3[0m:[93m53[0m
    [7m3[0m type _uniq_0 = { kind: "Ctor2" } | { kind: "Ctor3"; f0: _uniq_0 };
    [7m [0m [96m                                                    ~~[0m
    'f0' is declared here.
[96mstarcoderbase-7b/multitok/prog_16/var_6.ts[0m:[93m40[0m:[93m13[0m - [91merror[0m[90m TS2820: [0mType '"Ctor3"' is not assignable to type '"Ctor4" | "Ctor5"'. Did you mean '"Ctor4"'?

[7m40[0m       f0: { kind: "Ctor3", f0: { kind: "Ctor3", f0: { kind: "Ctor2" } } },
[7m  [0m [91m            ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_16/var_6.ts[0m:[93m5[0m:[93m7[0m
    [7m5[0m   | { kind: "Ctor4" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_16/var_6.ts[0m:[93m40[0m:[93m34[0m - [91merror[0m[90m TS2820: [0mType '"Ctor3"' is not assignable to type '"Ctor4" | "Ctor5"'. Did you mean '"Ctor4"'?

[7m40[0m       f0: { kind: "Ctor3", f0: { kind: "Ctor3", f0: { kind: "Ctor2" } } },
[7m  [0m [91m                                 ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_16/var_6.ts[0m:[93m5[0m:[93m7[0m
    [7m5[0m   | { kind: "Ctor4" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_16/var_6.ts[0m:[93m40[0m:[93m55[0m - [91merror[0m[90m TS2820: [0mType '"Ctor2"' is not assignable to type '"Ctor4" | "Ctor5"'. Did you mean '"Ctor4"'?

[7m40[0m       f0: { kind: "Ctor3", f0: { kind: "Ctor3", f0: { kind: "Ctor2" } } },
[7m  [0m [91m                                                      ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_16/var_6.ts[0m:[93m5[0m:[93m7[0m
    [7m5[0m   | { kind: "Ctor4" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_16/var_6.ts[0m:[93m49[0m:[93m13[0m - [91merror[0m[90m TS2820: [0mType '"Ctor3"' is not assignable to type '"Ctor4" | "Ctor5"'. Did you mean '"Ctor4"'?

[7m49[0m       f0: { kind: "Ctor3", f0: { kind: "Ctor2" } },
[7m  [0m [91m            ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_16/var_6.ts[0m:[93m5[0m:[93m7[0m
    [7m5[0m   | { kind: "Ctor4" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_16/var_6.ts[0m:[93m49[0m:[93m34[0m - [91merror[0m[90m TS2820: [0mType '"Ctor2"' is not assignable to type '"Ctor4" | "Ctor5"'. Did you mean '"Ctor4"'?

[7m49[0m       f0: { kind: "Ctor3", f0: { kind: "Ctor2" } },
[7m  [0m [91m                                 ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_16/var_6.ts[0m:[93m5[0m:[93m7[0m
    [7m5[0m   | { kind: "Ctor4" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_16/var_6.ts[0m:[93m52[0m:[93m15[0m - [91merror[0m[90m TS2820: [0mType '"Ctor3"' is not assignable to type '"Ctor4" | "Ctor5"'. Did you mean '"Ctor4"'?

[7m52[0m         f0: { kind: "Ctor3", f0: { kind: "Ctor2" } },
[7m  [0m [91m              ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_16/var_6.ts[0m:[93m5[0m:[93m7[0m
    [7m5[0m   | { kind: "Ctor4" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_16/var_6.ts[0m:[93m52[0m:[93m36[0m - [91merror[0m[90m TS2820: [0mType '"Ctor2"' is not assignable to type '"Ctor4" | "Ctor5"'. Did you mean '"Ctor4"'?

[7m52[0m         f0: { kind: "Ctor3", f0: { kind: "Ctor2" } },
[7m  [0m [91m                                   ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_16/var_6.ts[0m:[93m5[0m:[93m7[0m
    [7m5[0m   | { kind: "Ctor4" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_16/var_6.ts[0m:[93m63[0m:[93m13[0m - [91merror[0m[90m TS2820: [0mType '"Ctor3"' is not assignable to type '"Ctor4" | "Ctor5"'. Did you mean '"Ctor4"'?

[7m63[0m       f0: { kind: "Ctor3", f0: { kind: "Ctor2" } },
[7m  [0m [91m            ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_16/var_6.ts[0m:[93m5[0m:[93m7[0m
    [7m5[0m   | { kind: "Ctor4" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_16/var_6.ts[0m:[93m63[0m:[93m34[0m - [91merror[0m[90m TS2820: [0mType '"Ctor2"' is not assignable to type '"Ctor4" | "Ctor5"'. Did you mean '"Ctor4"'?

[7m63[0m       f0: { kind: "Ctor3", f0: { kind: "Ctor2" } },
[7m  [0m [91m                                 ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_16/var_6.ts[0m:[93m5[0m:[93m7[0m
    [7m5[0m   | { kind: "Ctor4" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_16/var_6.ts[0m:[93m80[0m:[93m13[0m - [91merror[0m[90m TS2820: [0mType '"Ctor2"' is not assignable to type '"Ctor4" | "Ctor5"'. Did you mean '"Ctor4"'?

[7m80[0m       f0: { kind: "Ctor2" },
[7m  [0m [91m            ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_16/var_6.ts[0m:[93m5[0m:[93m7[0m
    [7m5[0m   | { kind: "Ctor4" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_16/var_6.ts[0m:[93m83[0m:[93m15[0m - [91merror[0m[90m TS2820: [0mType '"Ctor3"' is not assignable to type '"Ctor4" | "Ctor5"'. Did you mean '"Ctor4"'?

[7m83[0m         f0: { kind: "Ctor3", f0: { kind: "Ctor2" } },
[7m  [0m [91m              ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_16/var_6.ts[0m:[93m5[0m:[93m7[0m
    [7m5[0m   | { kind: "Ctor4" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_16/var_6.ts[0m:[93m83[0m:[93m36[0m - [91merror[0m[90m TS2820: [0mType '"Ctor2"' is not assignable to type '"Ctor4" | "Ctor5"'. Did you mean '"Ctor4"'?

[7m83[0m         f0: { kind: "Ctor3", f0: { kind: "Ctor2" } },
[7m  [0m [91m                                   ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_16/var_6.ts[0m:[93m5[0m:[93m7[0m
    [7m5[0m   | { kind: "Ctor4" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_16/var_6.ts[0m:[93m86[0m:[93m17[0m - [91merror[0m[90m TS2820: [0mType '"Ctor3"' is not assignable to type '"Ctor4" | "Ctor5"'. Did you mean '"Ctor4"'?

[7m86[0m           f0: { kind: "Ctor3", f0: { kind: "Ctor3", f0: { kind: "Ctor2" } } },
[7m  [0m [91m                ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_16/var_6.ts[0m:[93m5[0m:[93m7[0m
    [7m5[0m   | { kind: "Ctor4" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_16/var_6.ts[0m:[93m86[0m:[93m38[0m - [91merror[0m[90m TS2820: [0mType '"Ctor3"' is not assignable to type '"Ctor4" | "Ctor5"'. Did you mean '"Ctor4"'?

[7m86[0m           f0: { kind: "Ctor3", f0: { kind: "Ctor3", f0: { kind: "Ctor2" } } },
[7m  [0m [91m                                     ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_16/var_6.ts[0m:[93m5[0m:[93m7[0m
    [7m5[0m   | { kind: "Ctor4" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_16/var_6.ts[0m:[93m86[0m:[93m59[0m - [91merror[0m[90m TS2820: [0mType '"Ctor2"' is not assignable to type '"Ctor4" | "Ctor5"'. Did you mean '"Ctor4"'?

[7m86[0m           f0: { kind: "Ctor3", f0: { kind: "Ctor3", f0: { kind: "Ctor2" } } },
[7m  [0m [91m                                                          ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_16/var_6.ts[0m:[93m5[0m:[93m7[0m
    [7m5[0m   | { kind: "Ctor4" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_16/var_6.ts[0m:[93m102[0m:[93m13[0m - [91merror[0m[90m TS2820: [0mType '"Ctor2"' is not assignable to type '"Ctor4" | "Ctor5"'. Did you mean '"Ctor4"'?

[7m102[0m       f0: { kind: "Ctor2" },
[7m   [0m [91m            ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_16/var_6.ts[0m:[93m5[0m:[93m7[0m
    [7m5[0m   | { kind: "Ctor4" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_16/var_6.ts[0m:[93m105[0m:[93m15[0m - [91merror[0m[90m TS2820: [0mType '"Ctor3"' is not assignable to type '"Ctor4" | "Ctor5"'. Did you mean '"Ctor4"'?

[7m105[0m         f0: { kind: "Ctor3", f0: { kind: "Ctor2" } },
[7m   [0m [91m              ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_16/var_6.ts[0m:[93m5[0m:[93m7[0m
    [7m5[0m   | { kind: "Ctor4" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_16/var_6.ts[0m:[93m105[0m:[93m36[0m - [91merror[0m[90m TS2820: [0mType '"Ctor2"' is not assignable to type '"Ctor4" | "Ctor5"'. Did you mean '"Ctor4"'?

[7m105[0m         f0: { kind: "Ctor3", f0: { kind: "Ctor2" } },
[7m   [0m [91m                                   ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_16/var_6.ts[0m:[93m5[0m:[93m7[0m
    [7m5[0m   | { kind: "Ctor4" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_16/var_6.ts[0m:[93m128[0m:[93m13[0m - [91merror[0m[90m TS2820: [0mType '"Ctor2"' is not assignable to type '"Ctor4" | "Ctor5"'. Did you mean '"Ctor4"'?

[7m128[0m       f0: { kind: "Ctor2" },
[7m   [0m [91m            ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_16/var_6.ts[0m:[93m5[0m:[93m7[0m
    [7m5[0m   | { kind: "Ctor4" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_16/var_6.ts[0m:[93m131[0m:[93m15[0m - [91merror[0m[90m TS2820: [0mType '"Ctor2"' is not assignable to type '"Ctor4" | "Ctor5"'. Did you mean '"Ctor4"'?

[7m131[0m         f0: { kind: "Ctor2" },
[7m   [0m [91m              ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_16/var_6.ts[0m:[93m5[0m:[93m7[0m
    [7m5[0m   | { kind: "Ctor4" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_16/var_6.ts[0m:[93m134[0m:[93m17[0m - [91merror[0m[90m TS2820: [0mType '"Ctor2"' is not assignable to type '"Ctor4" | "Ctor5"'. Did you mean '"Ctor4"'?

[7m134[0m           f0: { kind: "Ctor2" },
[7m   [0m [91m                ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_16/var_6.ts[0m:[93m5[0m:[93m7[0m
    [7m5[0m   | { kind: "Ctor4" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_16/var_6.ts[0m:[93m157[0m:[93m13[0m - [91merror[0m[90m TS2820: [0mType '"Ctor3"' is not assignable to type '"Ctor4" | "Ctor5"'. Did you mean '"Ctor4"'?

[7m157[0m       f0: { kind: "Ctor3", f0: { kind: "Ctor2" } },
[7m   [0m [91m            ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_16/var_6.ts[0m:[93m5[0m:[93m7[0m
    [7m5[0m   | { kind: "Ctor4" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_16/var_6.ts[0m:[93m157[0m:[93m34[0m - [91merror[0m[90m TS2820: [0mType '"Ctor2"' is not assignable to type '"Ctor4" | "Ctor5"'. Did you mean '"Ctor4"'?

[7m157[0m       f0: { kind: "Ctor3", f0: { kind: "Ctor2" } },
[7m   [0m [91m                                 ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_16/var_6.ts[0m:[93m5[0m:[93m7[0m
    [7m5[0m   | { kind: "Ctor4" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_16/var_6.ts[0m:[93m160[0m:[93m15[0m - [91merror[0m[90m TS2820: [0mType '"Ctor3"' is not assignable to type '"Ctor4" | "Ctor5"'. Did you mean '"Ctor4"'?

[7m160[0m         f0: { kind: "Ctor3", f0: { kind: "Ctor2" } },
[7m   [0m [91m              ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_16/var_6.ts[0m:[93m5[0m:[93m7[0m
    [7m5[0m   | { kind: "Ctor4" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_16/var_6.ts[0m:[93m160[0m:[93m36[0m - [91merror[0m[90m TS2820: [0mType '"Ctor2"' is not assignable to type '"Ctor4" | "Ctor5"'. Did you mean '"Ctor4"'?

[7m160[0m         f0: { kind: "Ctor3", f0: { kind: "Ctor2" } },
[7m   [0m [91m                                   ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_16/var_6.ts[0m:[93m5[0m:[93m7[0m
    [7m5[0m   | { kind: "Ctor4" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_16/var_6.ts[0m:[93m163[0m:[93m17[0m - [91merror[0m[90m TS2820: [0mType '"Ctor3"' is not assignable to type '"Ctor4" | "Ctor5"'. Did you mean '"Ctor4"'?

[7m163[0m           f0: { kind: "Ctor3", f0: { kind: "Ctor2" } },
[7m   [0m [91m                ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_16/var_6.ts[0m:[93m5[0m:[93m7[0m
    [7m5[0m   | { kind: "Ctor4" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_16/var_6.ts[0m:[93m163[0m:[93m38[0m - [91merror[0m[90m TS2820: [0mType '"Ctor2"' is not assignable to type '"Ctor4" | "Ctor5"'. Did you mean '"Ctor4"'?

[7m163[0m           f0: { kind: "Ctor3", f0: { kind: "Ctor2" } },
[7m   [0m [91m                                     ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_16/var_6.ts[0m:[93m5[0m:[93m7[0m
    [7m5[0m   | { kind: "Ctor4" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_16/var_6.ts[0m:[93m195[0m:[93m13[0m - [91merror[0m[90m TS2820: [0mType '"Ctor3"' is not assignable to type '"Ctor4" | "Ctor5"'. Did you mean '"Ctor4"'?

[7m195[0m       f0: { kind: "Ctor3", f0: { kind: "Ctor3", f0: { kind: "Ctor2" } } },
[7m   [0m [91m            ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_16/var_6.ts[0m:[93m5[0m:[93m7[0m
    [7m5[0m   | { kind: "Ctor4" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_16/var_6.ts[0m:[93m195[0m:[93m34[0m - [91merror[0m[90m TS2820: [0mType '"Ctor3"' is not assignable to type '"Ctor4" | "Ctor5"'. Did you mean '"Ctor4"'?

[7m195[0m       f0: { kind: "Ctor3", f0: { kind: "Ctor3", f0: { kind: "Ctor2" } } },
[7m   [0m [91m                                 ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_16/var_6.ts[0m:[93m5[0m:[93m7[0m
    [7m5[0m   | { kind: "Ctor4" }
    [7m [0m [96m      ~~~~[0m
    The expected type comes from property 'kind' which is declared here on type '_uniq_1'
[96mstarcoderbase-7b/multitok/prog_16/var_6.ts[0m:[93m195[0m:[93m55[0m - [91merror[0m[90m TS2820: [0mType '"Ctor2"' is not assignable to type '"Ctor4" | "Ctor5"'. Did you mean '"Ctor4"'?

[7m195[0m       f0: { kind: "Ctor3", f0: { kind: "Ctor3", f0: { kind: "Ctor2" } } },
[7m   [0m [91m                                                      ~~~~[0m

  [96mstarcoderbase-7b/multitok/prog_16/var_6.ts[0m:[93m5[0m:[93m7[0m
    [7m5[0m   | { kind: "Ctor4" }
    [7m [0m [96m      ~~~~[0m
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
    2339, 2345, 2820, 2820, 2820,
    2820, 2820, 2820, 2820, 2820,
    2820, 2820, 2820, 2820, 2820,
    2820, 2820, 2820, 2820, 2820,
    2820, 2820, 2820, 2820, 2820,
    2820, 2820, 2820, 2820, 2820,
    2820, 2820
  ]
}
