# Installation

TS install for boa
```
// install nvm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.35.3/install.sh | bash
// run nvm bash script
source ~/.bashrc
// install node
nvm install node
// install ts-node
mkdir ~/.npm_packages
npm install --prefix ~/.npm_packages ts-node typescript '@types/node'
npx --cache ~/.npm_packages ts-node --typeCheck <prog.ts>
```

Build package
```
python -m build
export PYTHONPATH=/path/to/your/package:$PYTHONPATH
```
In jupyer `kernel.json` add env var, then restart vscode.
```
"env": {"PYTHONPATH": "/path/to/your/package:$PYTHONPATH"}`
```

# TODO

- finish unittests

- Steering test split typecheck [MAKE IT ALWAYS SAME FOR SAME CANDIDATES]
- add limit 3000 to mutations [CLI ARG], 100 TYPECHECKED

- Run chat codellama on delta
- upload completion for SC1 and SC7
