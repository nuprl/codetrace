# Explanation

This dir contains variations of one ts program "date" from stenotype-eval-dataset.

- date_raw.ts: original program
- date.json: full metadata
- date_fim_DateQuery: fim'd type DateQuery, should be "medium" difficulty
- date_fim_ValidDateQuery: fim'd type ValidDateQuery, should be "easy" difficulty

## LLM performance

CodeLlama: high
Starcoder: high
single token problem: we can just assume that for correctness `Date ≡ DateQuery` and `ValidDateQuery ≡ Query` (although not quite true because of `new Date()`)