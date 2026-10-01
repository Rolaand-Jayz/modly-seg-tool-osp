# Project owned PBR estimation node

This Modly process node is a provisional CPU implementation of
**estimate-pbr-properties**. It consumes the exact input mesh and Structured Asset,
valid Ticket 06 material-region mappings, calibrated training RGB views, known
lights and cameras, and a per-pixel face/barycentric correspondence bundle. It
uses the project-owned v2 inverse-render estimator and keeps the source geometry
and topology unchanged.

`albedo_region_prior_strength` is bounded to `[0,10]` and defaults to `0` to
preserve the unregularized behavior. Development experiments used `1.0`; this
parameter, the v2 estimator digest, and its full parameter set are recorded in
stage provenance and input identity. No default is promoted based on development
scores alone.

The candidate estimates only linear base color, roughness, and metallic.
Unobserved texels remain unknown. Bump/height, tangent-space normal, opacity,
and emissive values are not synthesized. Training-fit residuals are recorded
as uncalibrated confidence, never as probability. The frozen Ticket 08 map and
novel-light thresholds and the RX 7900 GRE runtime gate are still required for
acceptance.

## Observation bundle

Set the workflow node's observation_bundle_path to a workspace-relative JSON
file containing these exact fields: schema, topology_revision, archive_path,
source_observation_ids, view_ids, and training_lights.

Use schema value **modly.ticket08.pbr-training-bundle.v1**. The
topology_revision must equal the Structured Asset revision. archive_path points
to a workspace-relative NPZ file. The source-observation IDs and view IDs are
parallel arrays: each index binds one training view to one observation
registered on the Structured Asset. training_lights is an array of records with
direction (three values) and linear RGB radiance (three nonnegative values).

The array archive is created with numpy.savez_compressed and contains only
numeric members readable with allow_pickle=False:

- face_ids: integer V x H x W; -1 means no visible mesh correspondence.
- barycentric: floating V x H x W x 3; entries follow each indexed face's
  vertex order. Unmapped pixels may contain NaN.
- face_uvs: floating F x 3 x 2; exactly the UV triangles decoded from the
  unchanged input mesh and in the same face order as Ticket 06.
- observations_linear: floating V x H x W x 3, calibrated linear RGB in
  [0,1].
- visible_masks: boolean V x H x W; every true pixel needs a valid face ID
  and finite barycentrics.
- camera_to_world: floating V x 4 x 4; column 2 of each matrix is the
  normalized direction from the visible surface toward that camera.

V must equal the number of registered source observations. Every face ID,
barycentric mapping, UV triangle, source digest, and topology revision is
validated before optimization. The JSON contract has no held-out-light field;
held-out observations belong only in the independent scorer.

The process writes a content-addressed NPZ map artifact, a stage-evidence JSON
artifact, and a new Structured Asset sidecar under StructuredAssets/. It
preserves unrelated assertions, corrections, and stage artifacts. Invalid
calibration, mismatched geometry/topology, absent Ticket 06 evidence, or
unregistered observation provenance stops the node with a bounded process
error and does not produce a successful Structured Asset result.
