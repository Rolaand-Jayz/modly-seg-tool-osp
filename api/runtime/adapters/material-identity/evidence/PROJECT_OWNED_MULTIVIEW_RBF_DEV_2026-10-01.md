# Project-owned four-view RBF material candidate development result

## Decision

The preregistered candidate is rejected at the development boundary. The 48
fixed combinations of four-view RBF settings and unknown-head scales produced
no candidate meeting all five unchanged Ticket 07 development gates. No model
weights were promoted, no held-out data was read, and no target-GPU inference
was run.

## Method and result

The run used one mean-plus-population-standard-deviation feature vector per
material region from its four topology-masked views. It trained and evaluated
with five object-disjoint folds using the fixed gamma/ridge/unknown-scale grid
in the preregistration. It used the existing 140-row development feature
artifact only, no accelerator, and the existing development-only label plan.

The strongest joint-gate-ratio candidate used gamma `1/60`, ridge `0.1`, and
unknown-head scale `0`. Its development results were:

- Macro-F1: `0.87727` (gate `>=0.85`, pass)
- Minimum supported-class recall: `0.60` (gate `>=0.80`, fail; clear plastic)
- Supported coverage: `0.96` (gate `>=0.87`, pass)
- Unknown abstention recall: `0.80` (gate `>=0.90`, fail)
- Ambiguous abstention recall: `1.00` (gate `>=0.90`, pass)
- Feasible threshold pairs: `0`

Clear plastic and glass remained the supported-label confusion; one of five
unknown regions remained a named painted-surface prediction. Across all 12
gamma/ridge combinations, the highest observed minimum supported-class recall
was `0.60`; no setting cleared every gate. The selected setting assigns zero
weight to the unknown head, so this result does not establish a useful unknown
head policy.

## Reproducibility and boundaries

Candidate source SHA-256:
`2b469ad49ac4e8ed4b6cd90cd9efc5d00aac15ae6816eade390a4b4eec96f65a`.
Preregistration: `PROJECT_OWNED_MULTIVIEW_RBF_PREREGISTRATION_2026-10-01.md`.
Input feature digest:
`7a0690741e6f9a5378bf934f02bca02588d6605f2c36d702e5ecf959b326040c`.
Report SHA-256:
`b27c41594f330e494d5b25cfed58bc751e389f98941c817a8a1ec7e9c4151778`.
Selected raw OOF output SHA-256:
`38a500ac22a33f4343af3eb0c17bfbec479b5835c8db72ee5e4822d34cd7bdff`.

The full report and raw development-only scores remain under the ignored
`.modly-amd-runtime/ticket07-project-owned-multiview-rbf-dev-20261001/` path.
The focused candidate tests pass `2/2` with
`python3 -m unittest api.tests.test_ticket07_multiview_rbf_candidate -v`;
Python compilation passed. These test code integrity and fold isolation, not
material accuracy. The frozen Ticket 07 gates remain unchanged and acceptance
is blocked.
