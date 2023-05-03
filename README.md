## Dev notes

## Need to be able to load:

- models from HF-HUB
- downloaded LLM checkpoints eg. Llama
- Santacoder 15B? checkpoints?

Note: no CPU load for now


## For starters:

- codegen
- santacoder
- llama
- galactica
- gpt-neox
- CodeGeeX

## Err log:

from GPT2 docs: the `attention_mask` always has to have the length `len(past_key_values) + len(input_ids)`