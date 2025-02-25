# Codetrace

This repo contains the code for the paper [Understanding How CodeLLMs (Mis)Predict Types with Activation Steering](https://arxiv.org/abs/2404.01903). Please cite as:

```bibtex
@article{lucchetti2024understanding,
  title={Understanding How CodeLLMs (Mis) Predict Types with Activation Steering},
  author={Lucchetti, Francesca and Guha, Arjun},
  journal={arXiv preprint arXiv:2404.01903},
  year={2024}
}
```

### Prerequisites

This repo requires [tree-sitter](https://tree-sitter.github.io/tree-sitter/), [vllm](https://github.com/vllm-project/vllm) and [nnsight](https://nnsight.net/). We used separate environments for each (see requirements file)
but for newer versions this is no longer necessary.

## Installation

TypeScript is required for typechecking. To install TS:

```bash
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

To build the package:
```bash
python -m build
export PYTHONPATH=/path/to/your/package:$PYTHONPATH
```

To add environment to jupyter kernel, in jupyer `kernel.json` add env var, then restart vscode.
```bash
"env": {"PYTHONPATH": "/path/to/your/package:$PYTHONPATH"}`
```