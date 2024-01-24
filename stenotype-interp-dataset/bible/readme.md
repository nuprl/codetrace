# Explanation

This dir contains variations of one ts program "bible" from stenotype-eval-dataset.

- bible_raw.ts: original program
- bible.json: full metadata
- bible_fim_unknown: fim'd type unknown, should be "hard" difficulty
- bible_fim_BookTitle: fim'd type BookTitle, should be "hard" difficulty

## LLM performance

CodeLlama: 0
Starcoder: good
`unknown` is hard because it is a broad definition. Surprisingly, `BookTitle` is also hard for CodeLlamas, probably because it is confusing the theme "bible". Starcoder finds both easy.