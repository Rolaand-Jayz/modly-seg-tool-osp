# Project-owned four-view RBF material candidate preregistration

Status: frozen before the development scoring run. This is a CPU-only
development candidate screen; it does not replace the current adapter, alter
the material ontology, or change any Ticket 07 gate.

## Candidate and fixed search

Each of the four topology-masked view descriptors for a single region is
combined into one 60-value signature: the mean and population standard
deviation of the existing 30 appearance cues. Training uses one row per
object/region, so repeated views cannot cross the fold boundary. A standardized
RBF kernel-ridge classifier has five supported material outputs and one
explicit unknown output. Ambiguous training examples have the existing uniform
supported-class target; they do not become a seventh forced class.

The full candidate grid is fixed in advance: gamma `{1/120, 1/60, 1/30,
1/15}`, ridge `{0.1, 1, 10}`, and unknown-head subtraction scale `{0, 0.5,
1, 2}`. There are five fixed object-disjoint folds. Each candidate's thresholds
are selected only from its development out-of-fold predictions, using the
existing calibration routine and unchanged joint-gate ordering. No held-out
rows, labels, logits, or truth are read.

The candidate implementation SHA-256 is
`2b469ad49ac4e8ed4b6cd90cd9efc5d00aac15ae6816eade390a4b4eec96f65a`.
Calibration source identities are `project_owned_dev_evaluator.py`
`98fdd637028e5cf9882f64945ffb5084522f550f6880851be6386957f3a1ebce`,
`dinov2_evaluator.py`
`2abcfc0cbb7a28f709f1bde1fd6884b571e5098e6f456cd574b29d38981e3aee`, and
`evaluator.py`
`c61e25d5baf8866414c9667bb03853ff93a58918cd85ec56df6d7d7792bbbb18`.

## Pinned data and unchanged limits

The candidate may read only the truth-free 140-row, 35-object development
feature artifact, SHA-256
`7a0690741e6f9a5378bf934f02bca02588d6605f2c36d702e5ecf959b326040c`. Its
fixture manifest is `c7c5d9c2ab31d3d956411c019e2ac19a72f991da5829ed18ceacf36e47094466`,
inputs are `453ac070aadd84eaf45eb381fb2910e906026b47cc929a5e6fdc4d6dd5f29694`,
and renderer source is
`6b654cbb31af7e598561a9f6516b807ba7281f77157281f76be90edb5aa6850c`.
Truth labels are joined only from the existing deterministic development-only
object plan. The frozen gate source SHA-256 is
`bbdb9e384fcc96f7abbbb33d92d09f37f092f18b4fe32f8a5744aa1eb85c5ca0`.

The candidate must jointly meet macro-F1 `>=0.85`, minimum supported-class
recall `>=0.80`, supported coverage `>=0.87`, unknown abstention recall
`>=0.90`, and ambiguous abstention recall `>=0.90`. If it fails development,
no held-out score or target-GPU run follows, and no candidate weights are
promoted.

The new output directory is
`.modly-amd-runtime/ticket07-project-owned-multiview-rbf-dev-20261001/`.
The artifact inputs and candidate code are read-only for the run; only this
new result directory is written.
