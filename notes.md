Stracoderabse-1b uch more vulnerable to trivial "confusing" edits of var names than 7b, 15b

# using gpt_bigcode in nnsight

to use Gpt_bigcode models in nnsight do this: after line 997 of modeling_gpt_bigcode.py add 

```
if len(position_ids.shape) == 1:
    position_ids = position_ids.reshape(1, position_ids.shape[0])
    ## repeat for batch_size
    position_ids = position_ids.expand(batch_size, -1)
```

Note this is a temporary hack and I have no idea if this breaks anything else
and remove `@torch.jit.script`