# Synthetic car development fixture

This directory contains a small, deterministic car-like mesh and per-face
category labels for local development of car-oriented review and seam editing.
It is made only from simple geometric primitives. It is not a real car, an
image reconstruction, output from a generation model, or Ticket 04/13
acceptance evidence. No third-party asset or source image is included.

Regenerate the GLB, labels, and manifest from the recipe with the API test
environment:

```sh
.modly-amd-runtime/api-test-venv/bin/python api/runtime/adapters/parts/fixtures/car-development/generate_car_fixture.py
```

The face-label list follows the triangle order after reloading the emitted GLB
with `trimesh.load(..., force="mesh", process=False)`. `topology_revision`
binds face membership to that loader's face-index order. `geometry_digest`
binds the labels to the oriented triangle coordinates. Regeneration may be
checked with `api.tests.test_car_development_fixture`.

The mesh is intentionally simple and has overlapping primitive components; it
is suitable for exercising topology-bound selection, face-region edits, and
viewer interaction, but it is not a realistic automotive segmentation
benchmark. It does not demonstrate inference, correction learning, or any
production model behavior.
