# Locating type knowledge in Code LLMs

Main notebook is `trace-demo`. 

## Models

- starcoder

## Experiments

1. head attribution for type detection task

### TODO

- crawl through `java` type examples from `MultiPLE`
- set up trace for head attribution using `baukit`

## Starcoder Notes

- no trail whitespaces
- set `clean_up_tokenization_spaces=False`
- full precision (no half), or float.b16
- uses `MQA` attention (significant) and flash attention (less significant?)

