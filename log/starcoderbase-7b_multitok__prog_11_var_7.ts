/home/franlucc/.npm_packages/_npx/1bf7c3c15bf47d04/node_modules/ts-node/src/index.ts:859
    return new TSError(diagnosticText, diagnosticCodes, diagnostics);
           ^
TSError: ⨯ Unable to compile TypeScript:
[96mstarcoderbase-7b/multitok/prog_11/var_7.ts[0m:[93m42[0m:[93m30[0m - [91merror[0m[90m TS2345: [0mArgument of type '_uniq_0' is not assignable to parameter of type '_uniq_1'.
  Type '{ kind: "Ctor2"; }' is not assignable to type '_uniq_1'.
    Type '{ kind: "Ctor2"; }' is missing the following properties from type '{ kind: "Ctor6"; f0: _uniq_1; f1: _uniq_1; }': f0, f1

[7m42[0m       return _uniq_7(_uniq_9(_uniq_18), _uniq_9(_uniq_19));
[7m  [0m [91m                             ~~~~~~~~[0m
[96mstarcoderbase-7b/multitok/prog_11/var_7.ts[0m:[93m47[0m:[93m30[0m - [91merror[0m[90m TS2345: [0mArgument of type '_uniq_0 | _uniq_1' is not assignable to parameter of type '_uniq_1'.
  Type '{ kind: "Ctor2"; }' is not assignable to type '_uniq_1'.

[7m47[0m       return _uniq_8(_uniq_9(_uniq_20), _uniq_9(_uniq_21));
[7m  [0m [91m                             ~~~~~~~~[0m
[96mstarcoderbase-7b/multitok/prog_11/var_7.ts[0m:[93m58[0m:[93m13[0m - [91merror[0m[90m TS2345: [0mArgument of type '{ kind: "Ctor5"; f0: { kind: "Ctor4"; f0: { kind: "Ctor3"; f0: { kind: "Ctor3"; f0: { kind: "Ctor3"; f0: { kind: "Ctor2"; }; }; }; }; }; f1: { kind: "Ctor4"; f0: { kind: "Ctor3"; f0: { kind: "Ctor3"; f0: { kind: "Ctor3"; f0: { kind: "Ctor3"; f0: { ...; }; }; }; }; }; }; }' is not assignable to parameter of type '_uniq_1'.
  Types of property 'f0' are incompatible.
    Type '{ kind: "Ctor4"; f0: { kind: "Ctor3"; f0: { kind: "Ctor3"; f0: { kind: "Ctor3"; f0: { kind: "Ctor2"; }; }; }; }; }' is not assignable to type '_uniq_0'.
      Types of property 'kind' are incompatible.
        Type '"Ctor4"' is not assignable to type '"Ctor2" | "Ctor3"'. Did you mean '"Ctor2"'?

[7m 58[0m     _uniq_9({
[7m   [0m [91m            ~[0m
[7m 59[0m       kind: "Ctor5",
[7m   [0m [91m~~~~~~~~~~~~~~~~~~~~[0m
[7m...[0m 
[7m 76[0m       },
[7m   [0m [91m~~~~~~~~[0m
[7m 77[0m     }),
[7m   [0m [91m~~~~~[0m
[96mstarcoderbase-7b/multitok/prog_11/var_7.ts[0m:[93m176[0m:[93m13[0m - [91merror[0m[90m TS2345: [0mArgument of type '{ kind: "Ctor5"; f0: { kind: "Ctor4"; f0: { kind: "Ctor3"; f0: { kind: "Ctor3"; f0: { kind: "Ctor3"; f0: { kind: "Ctor3"; f0: { kind: "Ctor2"; }; }; }; }; }; }; f1: { kind: "Ctor4"; f0: { kind: "Ctor3"; f0: { kind: "Ctor3"; f0: { ...; }; }; }; }; }' is not assignable to parameter of type '_uniq_1'.
  Types of property 'f0' are incompatible.
    Type '{ kind: "Ctor4"; f0: { kind: "Ctor3"; f0: { kind: "Ctor3"; f0: { kind: "Ctor3"; f0: { kind: "Ctor3"; f0: { kind: "Ctor2"; }; }; }; }; }; }' is not assignable to type '_uniq_0'.
      Types of property 'kind' are incompatible.
        Type '"Ctor4"' is not assignable to type '"Ctor2" | "Ctor3"'. Did you mean '"Ctor2"'?

[7m176[0m     _uniq_9({
[7m   [0m [91m            ~[0m
[7m177[0m       kind: "Ctor5",
[7m   [0m [91m~~~~~~~~~~~~~~~~~~~~[0m
[7m...[0m 
[7m197[0m       },
[7m   [0m [91m~~~~~~~~[0m
[7m198[0m     }),
[7m   [0m [91m~~~~~[0m

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
  diagnosticCodes: [ 2345, 2345, 2345, 2345 ]
}
