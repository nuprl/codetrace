# Installation

Make sure python 3.8

TS install for discovery - not needed ATM
```
mkdir ~/.npm_packages
npm install --prefix ~/.npm_packages ts-node typescript '@types/node'
npx --cache ~/.npm_packages ts-node
```

# Overview

Task: FIM type-inference
LLMs: CodeLlama 7b and 13b, starcoderbase

- dataset: contains TS programs for benchmark
- generations: contains single-token and multi-token LLM generations for TS programs in benchmark
- test_dataset.ipynb: testing benchmark against LLMs) to find easy/hard programs
- explore_ts_data.ipynb: looking through scraped TS dataset for benchmark problems