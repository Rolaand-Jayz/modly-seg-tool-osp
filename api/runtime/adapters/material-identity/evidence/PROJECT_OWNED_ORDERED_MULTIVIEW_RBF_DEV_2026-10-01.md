# Ordered four-view material candidate development result

## Decision

The candidate is rejected at the development boundary. The 48 preregistered
RBF and unknown-head settings performed worse than the unordered four-view
candidate. No model was promoted, no held-out rows or truth were accessed, and
no GPU inference was run.

## Result

The strongest candidate by the preregistered joint-gate ranking used gamma
`1/84`, ridge `10`, and unknown-head scale `0.5`. It measured:

- Macro-F1: `0.65286` (gate `>=0.85`, fail)
- Minimum supported-class recall: `0.40` (gate `>=0.80`, fail; clear plastic
  and glass)
- Supported coverage: `0.92` (gate `>=0.87`, pass)
- Unknown abstention recall: `0.40` (gate `>=0.90`, fail)
- Ambiguous abstention recall: `0.80` (gate `>=0.90`, fail)
- Feasible threshold pairs: `0`

Clear plastic and glass were repeatedly confused with one another; three of
five unknown development regions were named as painted surface. Preserving the
ordered highlight profile did not improve the candidate. No threshold, label,
or fixture was changed.

## Reproducibility and boundaries

Candidate source SHA-256:
`3713ae34dd150b44d035007eeab1fb75d670b308fadfac5820ed2cd3eb3d5723`.
Preregistration:
`PROJECT_OWNED_ORDERED_MULTIVIEW_RBF_PREREGISTRATION_2026-10-01.md`.
Pinned input digest:
`7a0690741e6f9a5378bf934f02bca02588d6605f2c36d702e5ecf959b326040c`.
Report SHA-256:
`d52e2ee9c1f769d4613d4cea12cf903d8e5ca567554d3e37b61846c1fa888de3`.
Selected raw OOF output SHA-256:
`ccf9514c269a23aec5c88f11b101845e0fc782a69b9801e53280704ca68c8f01`.

The full report remains in the ignored
`.modly-amd-runtime/ticket07-project-owned-ordered-multiview-rbf-dev-20261001/`
folder. The focused candidate tests pass `2/2` with
`python3 -m unittest api.tests.test_ticket07_multiview_ordered_rbf_candidate
-v`, and Python compilation passed. These checks cover signature ordering and
input rejection, not classifier accuracy. Ticket 07 remains blocked by the
unchanged quality gates.
