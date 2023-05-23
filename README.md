# Locating type knowledge in Code LLMs

Main notebook is `trace-demo`. Subdir `myrome` is modified ROME code to support bigcode models.

## Models

- starcoder

## Model Notes

- no trail whitespaces
- full precision (no half), or float.b16



## Eval performance on tasks

- complete var init with correct value according to type
    - vary context size
    - composite types
- complete operation with correct type
- control flow / nesting

## Experiment RQs

- Quantify how robust is it under trivial refactoring changes (eg. var names, formatting)
- Quantify how robust under nesting/control flow changes