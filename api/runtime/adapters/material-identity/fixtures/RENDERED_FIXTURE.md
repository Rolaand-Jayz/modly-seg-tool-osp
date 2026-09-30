# Ticket07 deterministic rendered evaluation fixture

Status: **generated and structurally verified; no classifier quality result**.

The fixture was produced with project-owned CPU rendering code in
`render_fixture.py`, using only Python 3.12.14, NumPy 2.5.3, and the standard
library in `.modly-amd-runtime/api-test-venv`. It did not invoke Blender, the
user's custom install, a GPU, a model, network access, or third-party image or
mesh assets. The ignored generated bundle is
`.modly-amd-runtime/material-identity-fixture-v1/` (135,875,428 bytes).

## Rendering and fixture scope

The renderer constructs lat-long ellipsoid meshes (48 sectors × 32 latitude
steps), rasterizes calibrated orthographic views by analytic ray/ellipsoid
intersection, maps visible pixels to mesh face IDs, and shades the view in
linear RGB with GGX direct lighting, environment reflection, and dielectric
Fresnel/refraction/absorption for clear materials. It writes RGB observations,
region masks, and int32 face maps. Every view includes a row-major 4×4
world-to-clip matrix and digest. The implementation is a deterministic
synthetic evaluator renderer, not a general-purpose path tracer or production
scene renderer. Its objects are distinct parameterized ellipsoids, so this
fixture measures crop identity behavior on controlled shape/material renders;
it does not establish generalization to varied real-world object classes.

The fixture has 145 unique mesh/object IDs and 580 region-view examples:

| Cohort | Development | Held out |
|---|---:|---:|
| Rubber/latex, Glass, Plastic, clear, Paint/plaster/enamel, Metal | 5 objects/class × 4 views = 20 regions/class | 20 different objects/class × 4 views = 80 regions/class |
| Unknown/out of taxonomy | 5 objects × 4 views = 20 regions | 5 different objects × 4 views = 20 regions |
| Ambiguous multi-material | 5 objects × 4 views = 20 regions | 5 different objects × 4 views = 20 regions |

All views of one object stay in its split. Four unknown identities are
represented across five objects/cohort/split: two wood objects and one each of
ceramic, paper, and stone. Ambiguous inputs are one
topology-bound region containing visibly separate rubber and paint surfaces;
the truth manifest declares abstention. Metal records separate bare and
painted subcohorts (painted objects expose a narrow bare-metal rolled edge),
but scoring remains on the supported generic `Metal` label only. These
authored identity labels and recipe IDs exist only in `truth.json`; they are
not passed to the classifier inputs. `inputs.json` contains opaque case/object
IDs, topology/mesh digests, region face IDs, image/mask/face-map paths and
digests, and camera projection matrices. The stage is usable for later
candidate evaluation, but the generated RGB/textures may still fail the frozen
model metrics; no success is assumed.

The held-out selective-coverage denominator is all **440 held-out regions**:
400 supported plus 20 unknown plus 20 ambiguous. Unknown and ambiguous
abstention are measured separately on their 20-example cohorts, each against
the unchanged >=90% gate. A prediction counts toward coverage only when it is
a single accepted label; correctly abstained unknown/ambiguous examples do not
count as covered. The unchanged >=80% coverage gate therefore requires at
least 352 of 440 single-label outputs; an evaluator must not calculate
coverage on supported examples alone.

## Reproduction and integrity

From the repository root, in the project API test environment:

```sh
PYTHONPATH=api .modly-amd-runtime/api-test-venv/bin/python \
  api/runtime/adapters/material-identity/fixtures/render_fixture.py \
  --output .modly-amd-runtime/material-identity-fixture-v1
```

The source/runtime pins are SHA-256 `6b654cbb31af7e598561a9f6516b807ba7281f77157281f76be90edb5aa6850c`
for `render_fixture.py`, Python `3.12.14`, and NumPy `2.5.3`. The generated
`fixture-manifest.json` indexes and byte-hashes all 1,887 data files, including
each mesh, RGB render, region mask, face map, and the two separated manifests.
The manifest itself is bound by its SHA-256 sidecar.

Generated evidence:

- Fixture manifest SHA-256: `c7c5d9c2ab31d3d956411c019e2ac19a72f991da5829ed18ceacf36e47094466`
- Input manifest SHA-256: `453ac070aadd84eaf45eb381fb2910e906026b47cc929a5e6fdc4d6dd5f29694`
- Truth manifest SHA-256: `8ca5699c1f3f7d96c2d28b024df6a67db0d9a6d725fbbf2e2f6f604bede6e6e8`
- Total generated regions: 580 (440 heldout, 140 development); total objects: 145; data files indexed: 1,887.
- Focused validation: `PYTHONPATH=api .modly-amd-runtime/api-test-venv/bin/python -m unittest discover -s api/tests -p 'test_ticket07_rendered_fixture.py' -v` — **6 passed** in 50.469 seconds. Tests repeat complete bundle generation and compare every artifact digest; verify support/split independence, taxonomy/abstention truth isolation, projection and pixel-exact mask/face-map consistency, projection/input digests, mesh face ID ordering, safe manifest-based regeneration, and the top-level manifest sidecar.

The existing pinned-image SigLIP2 process integration was separately reported
by the parent as 10/10 passed in 18.687 seconds after its prompt taxonomy and
stage-artifact updates. Its prompt set now has the five frozen primary classes
plus two auxiliary metal subtype prompts. It remains CPU-only synthetic
process evidence. This 580-region fixture has not
been passed through SigLIP2, and no accuracy, coverage, RX 7900 GRE runtime,
MIGraphX parity, latency, or VRAM result is claimed here.
