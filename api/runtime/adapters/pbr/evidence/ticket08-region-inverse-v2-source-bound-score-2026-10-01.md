# Ticket 08 frozen region-inverse v2 score (2026-10-01)

## Decision

The project-owned region-inverse v2 candidate was rejected by its single frozen
quality score. No thresholds, parameters, renderer, or fixture were changed.
The candidate is not accepted for Ticket 08, and this CPU result does not meet
the RX 7900 GRE execution gate.

## Region-map provenance

The input map was derived from the actual Modly Ticket 06 `segment-material-regions`
stage output produced from the four allowed training views. The source segmenter
was `modly.provisional-rgb-appearance-clustering@builtin:1.0.0`; its quality is
unqualified and its face confidence values are uncalibrated. It is only a
candidate material-region map, not material identity evidence.

The source GLB and the frozen Ticket 08 GLB have the same SHA-256:
`2d3a8b5ac0bfe44a60c8a7d83ff5b2042a6fac3b715806ac2e454f19b43f52d3`. Both
contain eight vertices and six indexed triangles. The Modly source sidecar used
topology revision `sha256:33b17ed2e206a5f311d6b9271af93801cb55934d4731be9c378546a680beafff`,
while the frozen correspondence contract uses
`sha256:77922db4079e22f05d6649baa6483218559ca8ccb18ab16c9d51b32ff03315c2`.
The map was explicitly rebound to the frozen revision only after exact GLB-byte
identity and the frozen accessor topology were checked. Original face indices
were carried unchanged; region IDs were recomputed using the frozen revision,
source label, and face IDs. The separate runtime audit records the full checks.

Ticket 06 stage bytes SHA-256:
`ce4b67d7ccb71c9a3b6cdeb4f592640d3cf2388c0f02b0e7ac1ce656b055b8d6`.
The source map contains three non-overlapping regions covering all six faces.
The resulting frozen map SHA-256 is
`173ab9622f6df89e1f7ed228216f41467d6487a459ed238a7256aacaf358db6a`.

## Frozen candidate run

The runner opened only the frozen input's `training_observations` and
`training_view_masks` members, plus frozen GLB accessors, correspondence inputs,
training camera/light inputs, and this topology-bound map. It reported no target
or held-out access, zero accelerators, CPU execution, and peak host RSS of
209,580 KiB. The frozen parameters were resolution 96, 45 fit evaluations,
minimum 3 samples, 4,096 region sample cap, roughness prior weight 0.002, and
albedo region prior strength 1.0.

Candidate output SHA-256:
`ec644c99529005c355ac3085e65ed708f4c462f902a1ba2b7704220f86ef2744`.
Runner report SHA-256:
`8930c014009341b702d8922ff4a48506b2273f92a3a1d8cc0b0518cb60d83ba8`.

## One-shot score

The scorer verified the frozen runner report, source/fixture locks, output bytes,
and region-map digest before opening target members. It then accessed only the
frozen target-field allowlist. The terminal score report is stored with the
runtime run and cannot be rerun against the same output directory.

| Metric | Result | Frozen limit | Result |
|---|---:|---:|---|
| Base-color linear RGB MAE | 0.12485 | <= 0.08 | Fail |
| Base-color SSIM | 0.55152 | >= 0.85 | Fail |
| Roughness MAE | 0.38590 | <= 0.10 | Fail |
| Metallic MAE | 0.26881 | <= 0.10 | Fail |
| Conductor/dielectric metallic bias | 0.41947 | <= 0.08 | Fail |
| Held-out novel-light MAE | 0.04840 | <= 0.08 | Pass |
| Visible texel coverage | 1.0 | Informational | Not a gate |

Score report SHA-256:
`4e60349919ccbfb2e0607e0b9b24c3eb15094ebd359ca46e342f655cdd3075a1`.

## Lock correction and checks

The runner could not start initially because `FROZEN_FORWARD_MODEL_SHA256` had a
one-character transcription error. The lock now records the measured source
SHA-256 `bcb39942b4e62a416aabe9aed50ec03773028627872cd598ede90d9b2a31bd79`.
This restores the intended exact identity check and does not alter the frozen
candidate, parameters, or thresholds. The focused protocol suite passed 3/3
with the corrected lock.

Runtime details and full per-run digests remain under the ignored
`.modly-amd-runtime/ticket08-source-bound-map-20261001/` folder and are excluded
from Git publication.
