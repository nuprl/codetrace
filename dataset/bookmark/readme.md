# Explanation

This dir contains variations of one ts program "bookmark" from stenotype-eval-dataset.
Note that I edited the original bookmark program to delete some redundant examples. Confusing the LLM with long context is not the point.

- bookmark_raw.ts: original program
- bookmark_raw_v2.ts: reduced-context program
- bookmark.json: full metadata
- bookmark_fim_null: should be "easy-medium" difficulty as context gets longer; LLM needs to undertsand that views field can be null from context. EDIT: too easy

## LLM performance

On reduced-context example both LLMs perform well. 
Edit: even long-context both LLMs perform well.