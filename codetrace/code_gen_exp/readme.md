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


### Note

- nnsight multi token generation does not allow loops
    - this is because meta model loses track of token generations once it generated EOS token
    so, loops work only as long as you know the max_num_tokens to generate BEFORE eos token