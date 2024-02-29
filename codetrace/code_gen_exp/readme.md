## Code generation steering

Verify that the mutation technique works for code gen. Use python.

This helps prove our method does not just recover original variable names, but helps
localize the correct circuit. Also, not using train data for steering.

### Prelim experiment

- mutate HumanEval/mbpp with renaming [ ]
    - cut solutions in half [X]
- eval script from MultiplE [ ]

- run completions on renamed, observe lower accuracy
- steer

### Full experiment

- use MultiPLT python