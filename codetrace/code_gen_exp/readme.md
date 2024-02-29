## Code generation steering

Verify that the mutation technique works for code gen. Use python.

This helps prove our method does not just recover original variable names, but helps
localize the correct circuit. Also, not using train data for steering.

### Prelim experiment

- mutate HumanEval/mbpp with renaming [X]
    - cut solutions in half [X]
- eval script from MultiplE [X]
- run completions on renamed, observe lower accuracy [X]

- steer []

### Full experiment

- use MultiPLT python