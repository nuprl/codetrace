## Notes


In the dedup experiment, we try to see if  dedup the mutated tseering tensors of duplicate types improves the occlusion/hydra phenomenon.

For a positive result we expect to see:

- steering vectors tested on their fit datastes perform well, whereas the random ablations on the same perform very low.
This means the model is following the direction of steering tensors, and not just occlusion (because occlusion performs bad)

- mutated vectors on CAA perform well, better than the random baseline. This means mutated vectors aren't JUST doing occlusion.


If mutated vectors perform as well as random baseline, or marginally better, two things could be happening:

- mutated vectors just dont help steering on natural prompts and can't generalize
- mutated vectors help a bit, but they are on such a different bias that the model prefers occlusion overall

To investigate which one it is:
- look at logit lens for patched layers
Logit lens shows steering works on fit, but not on caa (same as random). Try DAS next.

To force the model to follow mutated vectors, try DAS. Keep in mind if this degrades performance it means mutated vectors just don't help.