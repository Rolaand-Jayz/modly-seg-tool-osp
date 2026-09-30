# Identify Part Semantics

This built-in Modly process node consumes an existing Structured Asset, its
digest-verified source observations, and topology-bound part segments. It
discovers predictors only through an installed Python entry point in the
`modly.semantic_adapters` group. A workflow request cannot supply predictions,
an executable path, or an import name.

An adapter must expose `ADAPTER_ID`, `MODEL_ID`, `WEIGHTS_ID`,
`WEIGHTS_DIGEST`, and `predict_jsonl(invocation)`. Before loading the adapter,
the node hashes every executable Python/native source listed by its installed
distribution; it rechecks the complete source set after invocation and before
commit. It validates the output header against that identity and invocation,
including the exact frozen ontology prompt digest, then validates each
prediction against a current part mapping, evidence scoped to that part,
confidence schema, and provenance before attachment. Every asset, including a
single-part asset, requires topology-bound per-part image evidence. When a
valid evidence manifest is already attached it is reused. Otherwise, the node
invokes Modly's pinned `derive-part-scoped-observations` producer from the
segmented Structured Asset and its verified GeoSAM2 render, topology map, and
camera metadata. It registers the producer's manifest, crops, and masks as
separate StageArtifacts and persists them with the semantic result. The
`org.modly.part-scoped-image-manifest` schema binds each crop and binary mask
to the exact face mapping, topology, source render view, camera metadata, and
derivation adapter. Asset-level images are never treated as part-scoped
evidence. Missing or invalid per-part evidence fails closed before adapter
invocation; newly derived artifacts are removed if semantic inference or
persistence fails.
Responses follow `predictor-response-v1.json`. Candidate labels remain separate
assertions. Adapters that emit open-vocabulary labels preserve their original
text beside optional normalization; closed-choice adapters preserve their
exact choice and frozen mapping as a typed source assertion. Competing
candidates, unknown/ambiguous states, evidence, and provenance are retained.
The stage artifact is content-addressed and the updated sidecar is written
atomically under a per-asset lock after source sidecar, geometry, observation,
and rendered-view digests are revalidated. Competing semantic assertions from
other adapters remain present across a rerun. Segmentation IDs and mappings are
invariant across the operation.

The Decider adapter is distributed as a code-only Python package. Install it
into a Python overlay visible to the process extension with
`scripts/install-decider-semantic-adapter.sh`; the command prints the immutable
overlay path. Add that path to `PYTHONPATH` in the same runtime that executes
this node. The installer verifies the Decider entry point and its asset and
native-harness locks before publishing the overlay. Installation happens on the
host before node execution; containerized process extensions must mount the
returned overlay read-only and add it to `PYTHONPATH`. Re-running the installer
returns the existing overlay only after rechecking its entry point and locks.

The pinned model snapshot remains a separate, locally staged asset. Set
`MODELS_DIR` to Modly's model root and `MODLY_API_DIR` to the API directory
provided to the process extension. The adapter checks every asset, runtime,
and source-lock digest before inference and requires the network to be
disabled. The installer does not fetch weights, change model identity, or
qualify semantic quality. Ticket 05 remains unaccepted until the frozen
development scoring gates and the dependent RX 7900 GRE execution evidence
pass; a successful package registration alone is not acceptance evidence.

## Vision resolver

The node uses local Decider 2B Vision as its only configured resolver. The
adapter remains behind Modly's installed semantic-adapter extension point and
is reported as experimental until its AMD runtime, answer-slot parity, and
unchanged Ticket 05 semantic gates pass. The node fails closed if Decider is
missing or an unsupported provider value is supplied; it does not switch to
another provider.
