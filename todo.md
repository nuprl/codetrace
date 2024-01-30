# TODO

Methods

- logit lens (with generation?)
    [X] logit lens no generation (top logit)
- attn viz
    [X]
- general patching from->to (with generation?)
    [below]

Experiments

[] ts dataset
    [] logit lens of wrong/right examples
    [] attn viz of wrong/right examples


## Notes

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

## Wrong vs correct examples

wrong examples can be patched with types from "correct" runs at later layers, eg.18
Question: is the LLM doing the wrong circuit, or it the correct circuit but wrong algorithm?
- if wrong circuit, we need to identify right one
- if correct circuit, identify the algorithm it's using

How to verify this?
- Can I patch "wrong" examples onto "wrong" examples
    - I can. This means LLM thinks it's doing the right thing, but it's not.

The idea is that by layer 14/18 there is a complete type "indicator". But we want to modify this indicator to be the "correct" type. 

What to do now?
- can ICL examples help?

We know the "correct" predictions are in the top 10 logits.
- do a pass@100 sample generation to verify this

Idea:
- find the incorrect algorithm it's using
- use that knowledge to steer the model to the correct algorithm

## Idea

mask distractors? 
- better dataset
    - **maybe revisit steno and add own renaming**



## Design

"prompt" class
- model
- to patch or to collect

or maybe "batch" class

```
batch1 = Batch(model, prompts)
batch2 = Batch(model, prompts)
with batch1.collect(layers=[2,3]) as collector:
    with batch2.patch(layers=[2], idx=[(<fim-suf>,<fim-pre>)]) as patcher:
        pather.patches = [2 : (2,3), 3 : [(3,4),(3,5)]]
        # do stuff
        activations = collector.activations
        patcher.patch(activations)
```
constraint: can only patch from same layer to same layer
- can patch any idx

don't waste too much time on this


## Today

[] better dataset - steno with own renaming
[] what is going on in "wrong" examples - what is the algorithm