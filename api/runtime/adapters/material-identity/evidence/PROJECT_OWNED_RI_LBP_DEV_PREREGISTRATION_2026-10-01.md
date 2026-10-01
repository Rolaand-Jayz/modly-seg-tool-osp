# Project-owned rotation-invariant texture candidate preregistration

Status: frozen before the Ticket 07 development-only run. This adds one
appearance representation candidate; it does not change labels, data splits,
acceptance gates, or the existing selected model.

## Candidate fixed in advance

- Feature representation: the existing 30 topology-masked physics and
  appearance cues followed by a 10-bin, rotation-invariant local binary
  pattern histogram. The histogram counts local 8-neighbor patterns by set
  bit-count for uniform patterns and places non-uniform patterns in one bin.
  Neighbor samples are used only when the entire 3x3 neighborhood remains in
  the material-region mask. The representation is implemented by
  `_rotation_invariant_texture_features` in
  `project_owned_classifier.py`.
- Classifier: the existing CPU RBF kernel ridge head with training-only
  unknown/ambiguous observations and mean pooling over the four views of each
  region. No explicit unknown head, synthetic training observations, or
  augmentation variants are enabled.
- Fixed settings: ridge `1.0`; RBF gamma `1/40`; seed `20261001`; zero
  augmentation variants; zero stress variants; one candidate and one
  development run.
- Development split only: the pinned 140-row, 35-object Ticket 07 development
  data and five object-disjoint folds. The evaluator does not open held-out
  rows or `truth.json`.

## Gates retained unchanged

The candidate must jointly meet macro-F1 `>=0.85`, minimum supported-class
recall `>=0.80`, supported coverage `>=0.87`, unknown abstention recall
`>=0.90`, and ambiguous abstention recall `>=0.90` on the development split.
All 20 unknown and all 20 ambiguous development rows remain in their frozen
object-disjoint cohorts. No gate can be traded against another or altered by
this candidate.

## Frozen inputs and source identities

- Fixture manifest: `c7c5d9c2ab31d3d956411c019e2ac19a72f991da5829ed18ceacf36e47094466`
- Fixture inputs: `453ac070aadd84eaf45eb381fb2910e906026b47cc929a5e6fdc4d6dd5f29694`
- Renderer source: `6b654cbb31af7e598561a9f6516b807ba7281f77157281f76be90edb5aa6850c`
- Candidate implementation: `1fcd4fa4d25b85fc0eb1d13d9ddb49c4eac6886a26b61eacd2d17b08075c4e5e`
- Development evaluator: `98fdd637028e5cf9882f64945ffb5084522f550f6880851be6386957f3a1ebce`
- Frozen gates document: `bbdb9e384fcc96f7abbbb33d92d09f37f092f18b4fe32f8a5744aa1eb85c5ca0`

The output directory is the new, empty ignored path
`.modly-amd-runtime/ticket07-project-owned-ri-lbp-dev-20261001/`. The exact
candidate specification and source digests in this file are fixed before that
directory is populated. No held-out results, model weights, external model, or
GPU are part of this run.

## Development result

The one fixed candidate was rejected at the development boundary. It produced
macro-F1 `0.8014652` (gate `>=0.85`), minimum class recall `0.40` (gate
`>=0.80`, glass), supported coverage `0.84` (gate `>=0.87`), unknown
abstention recall `0.40` (gate `>=0.90`), and ambiguous abstention recall
`1.00` (gate `>=0.90`). There were zero feasible threshold pairs. It accepted
12/20 unknown regions as `Paint/plaster/enamel`; all 20 ambiguous rows were
abstained as `unknown` rather than the exact `ambiguous` label. Held-out truth
was not opened and zero held-out rows were scored. No model artifact was
promoted.

The durable report is
`.modly-amd-runtime/ticket07-project-owned-ri-lbp-dev-20261001/report.json`
with SHA-256 `f654ebe07ab5207a638f1c1f8cd5702a3e829fd9351595a69e10769c0c91b937`;
the OOF logits SHA-256 is
`7af3de10fb771e2bd8803d6d91549efd2514d82523dea8683ee16cfceade7b61`. The
new descriptor is rotation-invariant in a synthetic unit test and the
40-dimensional model round-trips with provenance, but this candidate performs
worse than the existing mean-view RBF candidate and is not eligible for
promotion.
