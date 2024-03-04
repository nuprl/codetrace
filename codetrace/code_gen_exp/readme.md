## Code generation steering

Verify that the mutation technique works for code gen. Use python.

This helps prove our method does not just recover original variable names, but helps
localize the correct circuit. Also, not using train data for steering.

### Prelim experiment

- check renaming for errors []
- steer []
    - make more efficient by using nnsight (need dev branch for multi-tok generation)


### Note

- nnsight multi token generation does not allow loops
    - this is because meta model loses track of token generations once it generated EOS token
    so, loops work only as long as you know the max_num_tokens to generate BEFORE eos token

Using baukit. Question: what to steer? Last token only once. look at ITI


### Full experiment

- use MultiPLT python