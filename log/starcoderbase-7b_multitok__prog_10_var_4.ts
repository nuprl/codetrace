/home/franlucc/.npm_packages/_npx/1bf7c3c15bf47d04/node_modules/ts-node/src/index.ts:859
    return new TSError(diagnosticText, diagnosticCodes, diagnostics);
           ^
TSError: ⨯ Unable to compile TypeScript:
[96mstarcoderbase-7b/multitok/prog_10/var_4.ts[0m:[93m4[0m:[93m59[0m - [91merror[0m[90m TS2693: [0m'_uniq_1' only refers to a type, but is being used as a value here.

[7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6" }; f0: _uniq_1 };
[7m [0m [91m                                                          ~~~~~~~[0m
[96mstarcoderbase-7b/multitok/prog_10/var_4.ts[0m:[93m15[0m:[93m31[0m - [91merror[0m[90m TS2339: [0mProperty 'f0' does not exist on type '{ kind: "Ctor6"; }'.

[7m15[0m       let _uniq_13 = _uniq_11.f0;
[7m  [0m [91m                              ~~[0m
[96mstarcoderbase-7b/multitok/prog_10/var_4.ts[0m:[93m16[0m:[93m31[0m - [91merror[0m[90m TS2353: [0mObject literal may only specify known properties, and 'f0' does not exist in type '{ kind: "Ctor6"; }'.

[7m16[0m       return { kind: "Ctor6", f0: _uniq_9(_uniq_13, _uniq_12) };
[7m  [0m [91m                              ~~[0m
[96mstarcoderbase-7b/multitok/prog_10/var_4.ts[0m:[93m32[0m:[93m35[0m - [91merror[0m[90m TS2353: [0mObject literal may only specify known properties, and 'f0' does not exist in type '{ kind: "Ctor6"; }'.

[7m32[0m           return { kind: "Ctor6", f0: { kind: "Ctor5" } };
[7m  [0m [91m                                  ~~[0m
[96mstarcoderbase-7b/multitok/prog_10/var_4.ts[0m:[93m35[0m:[93m35[0m - [91merror[0m[90m TS2339: [0mProperty 'f0' does not exist on type '{ kind: "Ctor6"; }'.

[7m35[0m           let _uniq_19 = _uniq_15.f0;
[7m  [0m [91m                                  ~~[0m
[96mstarcoderbase-7b/multitok/prog_10/var_4.ts[0m:[93m51[0m:[93m50[0m - [91merror[0m[90m TS2353: [0mObject literal may only specify known properties, and 'f0' does not exist in type '{ kind: "Ctor6"; }'.

[7m51[0m     _uniq_10({ kind: "Ctor7" }, { kind: "Ctor6", f0: { kind: "Ctor5" } }),
[7m  [0m [91m                                                 ~~[0m
[96mstarcoderbase-7b/multitok/prog_10/var_4.ts[0m:[93m57[0m:[93m24[0m - [91merror[0m[90m TS2353: [0mObject literal may only specify known properties, and 'f0' does not exist in type '{ kind: "Ctor6"; }'.

[7m57[0m       { kind: "Ctor6", f0: { kind: "Ctor6", f0: { kind: "Ctor5" } } }
[7m  [0m [91m                       ~~[0m
[96mstarcoderbase-7b/multitok/prog_10/var_4.ts[0m:[93m66[0m:[93m9[0m - [91merror[0m[90m TS2353: [0mObject literal may only specify known properties, and 'f0' does not exist in type '{ kind: "Ctor6"; }'.

[7m66[0m         f0: { kind: "Ctor6", f0: { kind: "Ctor6", f0: { kind: "Ctor5" } } },
[7m  [0m [91m        ~~[0m
[96mstarcoderbase-7b/multitok/prog_10/var_4.ts[0m:[93m91[0m:[93m24[0m - [91merror[0m[90m TS2353: [0mObject literal may only specify known properties, and 'f0' does not exist in type '{ kind: "Ctor6"; }'.

[7m91[0m       { kind: "Ctor6", f0: { kind: "Ctor5" } }
[7m  [0m [91m                       ~~[0m
[96mstarcoderbase-7b/multitok/prog_10/var_4.ts[0m:[93m103[0m:[93m24[0m - [91merror[0m[90m TS2353: [0mObject literal may only specify known properties, and 'f0' does not exist in type '{ kind: "Ctor6"; }'.

[7m103[0m       { kind: "Ctor6", f0: { kind: "Ctor6", f0: { kind: "Ctor5" } } }
[7m   [0m [91m                       ~~[0m
[96mstarcoderbase-7b/multitok/prog_10/var_4.ts[0m:[93m117[0m:[93m9[0m - [91merror[0m[90m TS2353: [0mObject literal may only specify known properties, and 'f0' does not exist in type '{ kind: "Ctor6"; }'.

[7m117[0m         f0: { kind: "Ctor6", f0: { kind: "Ctor6", f0: { kind: "Ctor5" } } },
[7m   [0m [91m        ~~[0m
[96mstarcoderbase-7b/multitok/prog_10/var_4.ts[0m:[93m152[0m:[93m24[0m - [91merror[0m[90m TS2353: [0mObject literal may only specify known properties, and 'f0' does not exist in type '{ kind: "Ctor6"; }'.

[7m152[0m       { kind: "Ctor6", f0: { kind: "Ctor5" } }
[7m   [0m [91m                       ~~[0m
[96mstarcoderbase-7b/multitok/prog_10/var_4.ts[0m:[93m169[0m:[93m24[0m - [91merror[0m[90m TS2353: [0mObject literal may only specify known properties, and 'f0' does not exist in type '{ kind: "Ctor6"; }'.

[7m169[0m       { kind: "Ctor6", f0: { kind: "Ctor6", f0: { kind: "Ctor5" } } }
[7m   [0m [91m                       ~~[0m
[96mstarcoderbase-7b/multitok/prog_10/var_4.ts[0m:[93m188[0m:[93m9[0m - [91merror[0m[90m TS2353: [0mObject literal may only specify known properties, and 'f0' does not exist in type '{ kind: "Ctor6"; }'.

[7m188[0m         f0: { kind: "Ctor6", f0: { kind: "Ctor6", f0: { kind: "Ctor5" } } },
[7m   [0m [91m        ~~[0m
[96mstarcoderbase-7b/multitok/prog_10/var_4.ts[0m:[93m223[0m:[93m24[0m - [91merror[0m[90m TS2353: [0mObject literal may only specify known properties, and 'f0' does not exist in type '{ kind: "Ctor6"; }'.

[7m223[0m       { kind: "Ctor6", f0: { kind: "Ctor5" } }
[7m   [0m [91m                       ~~[0m
[96mstarcoderbase-7b/multitok/prog_10/var_4.ts[0m:[93m240[0m:[93m24[0m - [91merror[0m[90m TS2353: [0mObject literal may only specify known properties, and 'f0' does not exist in type '{ kind: "Ctor6"; }'.

[7m240[0m       { kind: "Ctor6", f0: { kind: "Ctor6", f0: { kind: "Ctor5" } } }
[7m   [0m [91m                       ~~[0m
[96mstarcoderbase-7b/multitok/prog_10/var_4.ts[0m:[93m259[0m:[93m9[0m - [91merror[0m[90m TS2353: [0mObject literal may only specify known properties, and 'f0' does not exist in type '{ kind: "Ctor6"; }'.

[7m259[0m         f0: { kind: "Ctor6", f0: { kind: "Ctor6", f0: { kind: "Ctor5" } } },
[7m   [0m [91m        ~~[0m
[96mstarcoderbase-7b/multitok/prog_10/var_4.ts[0m:[93m304[0m:[93m24[0m - [91merror[0m[90m TS2353: [0mObject literal may only specify known properties, and 'f0' does not exist in type '{ kind: "Ctor6"; }'.

[7m304[0m       { kind: "Ctor6", f0: { kind: "Ctor5" } }
[7m   [0m [91m                       ~~[0m
[96mstarcoderbase-7b/multitok/prog_10/var_4.ts[0m:[93m326[0m:[93m24[0m - [91merror[0m[90m TS2353: [0mObject literal may only specify known properties, and 'f0' does not exist in type '{ kind: "Ctor6"; }'.

[7m326[0m       { kind: "Ctor6", f0: { kind: "Ctor6", f0: { kind: "Ctor5" } } }
[7m   [0m [91m                       ~~[0m
[96mstarcoderbase-7b/multitok/prog_10/var_4.ts[0m:[93m350[0m:[93m9[0m - [91merror[0m[90m TS2353: [0mObject literal may only specify known properties, and 'f0' does not exist in type '{ kind: "Ctor6"; }'.

[7m350[0m         f0: { kind: "Ctor6", f0: { kind: "Ctor6", f0: { kind: "Ctor5" } } },
[7m   [0m [91m        ~~[0m
[96mstarcoderbase-7b/multitok/prog_10/var_4.ts[0m:[93m405[0m:[93m24[0m - [91merror[0m[90m TS2353: [0mObject literal may only specify known properties, and 'f0' does not exist in type '{ kind: "Ctor6"; }'.

[7m405[0m       { kind: "Ctor6", f0: { kind: "Ctor5" } }
[7m   [0m [91m                       ~~[0m
[96mstarcoderbase-7b/multitok/prog_10/var_4.ts[0m:[93m432[0m:[93m24[0m - [91merror[0m[90m TS2353: [0mObject literal may only specify known properties, and 'f0' does not exist in type '{ kind: "Ctor6"; }'.

[7m432[0m       { kind: "Ctor6", f0: { kind: "Ctor6", f0: { kind: "Ctor5" } } }
[7m   [0m [91m                       ~~[0m
[96mstarcoderbase-7b/multitok/prog_10/var_4.ts[0m:[93m461[0m:[93m9[0m - [91merror[0m[90m TS2353: [0mObject literal may only specify known properties, and 'f0' does not exist in type '{ kind: "Ctor6"; }'.

[7m461[0m         f0: { kind: "Ctor6", f0: { kind: "Ctor6", f0: { kind: "Ctor5" } } },
[7m   [0m [91m        ~~[0m
[96mstarcoderbase-7b/multitok/prog_10/var_4.ts[0m:[93m4[0m:[93m67[0m - [91merror[0m[90m TS1128: [0mDeclaration or statement expected.

[7m4[0m type _uniq_1 = { kind: "Ctor5" } | { kind: "Ctor6" }; f0: _uniq_1 };
[7m [0m [91m                                                                  ~[0m

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
    2693, 2339, 2353, 2353,
    2339, 2353, 2353, 2353,
    2353, 2353, 2353, 2353,
    2353, 2353, 2353, 2353,
    2353, 2353, 2353, 2353,
    2353, 2353, 2353, 1128
  ]
}
