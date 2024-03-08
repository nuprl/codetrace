# Installation

TS install for boa - not needed ATM
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

# Overview

Main interp and util code is in `codetrace/` toplevel. Experiments are organized in subdirectories. The REPL in each experiment subdirectory is `<exp_subdir>/main.ipynb`. Random notebooks in `notebooks/`. Note that only experiment subdir with `__init__.py` are up to date, all others deprecated.

# TODO

large steering experiments
- finalize py typeinf steering [X]
- finalize ts typeinf steering []