# Installation

Make sure python 3.8

TS install for discovery - not needed ATM
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

# Overview

Task: FIM type-inference
LLMs: CodeLlama 7b and 13b, starcoderbase

- dataset: contains TS programs for benchmark
- generations: contains single-token and multi-token LLM generations for TS programs in benchmark
- test_dataset.ipynb: testing benchmark against LLMs) to find easy/hard programs
- explore_ts_data.ipynb: looking through scraped TS dataset for benchmark problems