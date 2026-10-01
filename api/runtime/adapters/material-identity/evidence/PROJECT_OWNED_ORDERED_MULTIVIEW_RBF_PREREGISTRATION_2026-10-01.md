# Ordered four-view material feature candidate preregistration

Status: frozen before development scoring. This CPU-only development screen
does not change the Ticket 07 label set, production adapter, data split, or
quality gates.

## Fixed feature design and candidate grid

The input is the pinned truth-free 140-row development feature artifact. For
each region, four calibrated view rows are sorted by stable `view_id`. The
candidate signature contains (1) the mean and population standard deviation
of all 30 existing topology-masked cues (60 values), then (2) the six
highlight/contrast cues at indices 15, 16, 17, 18, 24, and 25 from each view in
that fixed order (24 values). Those six cues are highlight occupancy at two
luminance levels, luminance spread, saturation spread, bright-tail contrast,
and achromatic-highlight fraction. The final signature has 84 values.

The classifier and folds are fixed: standardized RBF kernel ridge, one
training row per object/region, five folds grouped by object identity, and an
explicit unknown output. Ambiguous training rows use a uniform target across
the five supported classes. The fixed grid contains gamma `{1/168, 1/84,
1/42, 1/21}`, ridge `{0.1, 1, 10}`, and unknown-head scales `{0, 0.5, 1, 2}`.
Threshold selection uses the existing development-only calibrator. No
held-out rows, truth, logits, or labels may be read.

Candidate implementation SHA-256:
`3713ae34dd150b44d035007eeab1fb75d670b308fadfac5820ed2cd3eb3d5723`.
The prior project-owned 30-cue extractor SHA-256 is
`1fcd4fa4d25b85fc0eb1d13d9ddb49c4eac6886a26b61eacd2d17b08075c4e5e`.
Development-plan/calibrator sources match their existing pinned hashes:
`dinov2_evaluator.py` `2abcfc0cbb7a28f709f1bde1fd6884b571e5098e6f456cd574b29d38981e3aee`,
`project_owned_dev_evaluator.py`
`98fdd637028e5cf9882f64945ffb5084522f550f6880851be6386957f3a1ebce`, and
`evaluator.py`
`c61e25d5baf8866414c9667bb03853ff93a58918cd85ec56df6d7d7792bbbb18`.

## Pinned input and unchanged gates

Feature artifact SHA-256:
`7a0690741e6f9a5378bf934f02bca02588d6605f2c36d702e5ecf959b326040c`.
Fixture manifest SHA-256:
`c7c5d9c2ab31d3d956411c019e2ac19a72f991da5829ed18ceacf36e47094466`.
Fixture input SHA-256:
`453ac070aadd84eaf45eb381fb2910e906026b47cc929a5e6fdc4d6dd5f29694`.
Renderer source SHA-256:
`6b654cbb31af7e598561a9f6516b807ba7281f77157281f76be90edb5aa6850c`.
The frozen gate file SHA-256 is
`bbdb9e384fcc96f7abbbb33d92d09f37f092f18b4fe32f8a5744aa1eb85c5ca0`.

Every candidate must jointly achieve macro-F1 `>=0.85`, minimum supported
class recall `>=0.80`, supported coverage `>=0.87`, unknown abstention recall
`>=0.90`, and ambiguous abstention recall `>=0.90`. A development failure
ends this candidate: no held-out scoring or target-GPU run follows and no
weights are promoted.

The only writable run path is the new ignored folder
`.modly-amd-runtime/ticket07-project-owned-ordered-multiview-rbf-dev-20261001/`.
