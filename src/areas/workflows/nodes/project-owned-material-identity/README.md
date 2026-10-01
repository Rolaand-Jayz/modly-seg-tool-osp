# Project-owned material identity candidate

This opt-in process node consumes a current Structured Asset, Ticket 06's
calibrated material-region view artifact, the source image observations, and an
explicit candidate JSON path. It projects the validated glTF triangles through
the exact stored Ticket 06 view matrices, intersects visible faces with each
saved material mask, and classifies only those topology-bound pixels.

The candidate file must be trained separately with
`api/runtime/adapters/material-identity/project_owned_classifier.py` and
explicitly supplied to the workflow node. The node does not train, alter
thresholds, or use saved corrections as training data. Uncalibrated weights
run only to emit explicit `unknown` abstentions with `uncalibrated` confidence
and `unqualified` provenance; they cannot produce a supported material
identity. Calibrated candidates may emit `ambiguous` abstentions, but a
calibrated threshold alone is not Ticket 07 acceptance. PBR assertions and
unrelated identity evidence are preserved.

The node is a CPU candidate implementation. Its deterministic tests establish
the process contract and face-to-source-image mapping on the Ticket 06 fixture;
they do not establish the Ticket 07 frozen quality metrics, held-out quality,
AMD performance, or release acceptance.
