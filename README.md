# Locating type knowledge in Code LLMs

Main notebook is `codetrace`. Subdir `myrome` is modified ROME code to support bigcode models.

## Models

- santacoder
- starcoder

## Model Notes

- no trail whitespaces
- full precision (no half)


## Eval performance on tasks

- complete var init with correct value according to type
    - vary context size
    - composite types
- complete operation with correct type
- control flow / nesting

## Experiment RQs

- Quantify how robust is it under trivial refactoring changes (eg. var names, formatting)
- QUantiyf how robust under nesting/control flow changes