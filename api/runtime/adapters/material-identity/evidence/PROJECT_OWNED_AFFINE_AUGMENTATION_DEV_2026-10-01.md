# Project-owned classifier affine augmentation development run — 2026-10-01

Status: **development-only candidate; frozen Ticket 07 gates failed**.

## Change

`project_owned_classifier.py` now augments training crops with a deterministic,
bounded 2D view-plane affine warp, applied jointly to RGB and the topology mask.
RGB is bilinearly sampled and the mask is nearest-neighbor sampled. The warp
varies rotation (±14°), scale (0.88–1.12), shear (±0.08), and crop translation
(±6%). Existing color, brightness, directional-lighting, low-frequency texture,
and sensor-noise changes remain. This is a 2D crop/framing approximation; it is
not a renderer-based new 3D view or new geometry.

## Evaluation

One fixed candidate was evaluated on the frozen 140-row development split with
five object-disjoint folds, one synthetic training variant per view, the frozen
development threshold search, and no synthetic stress rows. The classifier fit
uses only labels from the predeclared development truth plan. Raw development
features were written before that plan was joined. The held-out truth file was
not opened, zero held-out rows were read, and no GPU was used.

| Measure | Candidate | Frozen gate |
| --- | ---: | ---: |
| Supported material macro-F1 | 0.62755 | ≥ 0.85 |
| Minimum supported-class recall | 0.30 | ≥ 0.80 |
| Supported-region coverage | 0.69 | ≥ 0.87 |
| Unknown abstention recall | 0.40 | ≥ 0.90 |
| Ambiguous abstention recall | 1.00 | ≥ 0.90 |

The candidate had zero feasible threshold pairs and no provisional weights were
written. Its ambiguous exact-status recall was 0.00: all 20 ambiguous development
views were labeled `unknown`, which meets the frozen abstention metric but does
not produce the more informative exact `ambiguous` status. Per-class supported
recall was rubber 1.00, metal 0.85, glass 0.40, clear plastic 0.30, and painted
metal 0.35. The truth-free fixed-feature baseline in the same report scored
macro-F1 0.62610, minimum recall 0.40, supported coverage 0.69, unknown
abstention 0.45, and ambiguous abstention 0.45. The small change does not
establish a useful generalization improvement.

## Provenance

- Frozen fixture manifest SHA-256: `c7c5d9c2ab31d3d956411c019e2ac19a72f991da5829ed18ceacf36e47094466`
- Frozen inputs SHA-256: `453ac070aadd84eaf45eb381fb2910e906026b47cc929a5e6fdc4d6dd5f29694`
- Pinned renderer source SHA-256: `6b654cbb31af7e598561a9f6516b807ba7281f77157281f76be90edb5aa6850c`
- Classifier source SHA-256: `50f7a286a05776c187ad1b7ccbe61878aaf996e813152e45a7ff8a9c064ddf7e`
- Development evaluator source SHA-256 after correction to future-run augmentation wording: `4f641ad5327a1e4db8e3f195c6befd1b6ec87c748490aa66658ff21ca1790f94`
- Augmentation test source SHA-256: `a00c96f643caa7ebd9818958b4c9a0f74993fcebbca91d520c35dddaed910c0a`
- Development feature artifact SHA-256: `7a0690741e6f9a5378bf934f02bca02588d6605f2c36d702e5ecf959b326040c`
- Development OOF artifact SHA-256: `66b6c02a8ddeb1bc87ba79425cfb06018ec2ac05bde66882d996a0bedf4d44a2`
- Evaluation report SHA-256: `ff2a18dee85062fab3e0812b4fd16785b17c6cdd19788ecb2e3b3227af97e7af`
- Full raw features, OOF rows, and report are in the ignored runtime directory
  `.modly-amd-runtime/ticket07-project-owned-view-affine-20261001-v1/`.

Focused CPU checks:

```text
python3 -m unittest \
  api.tests.test_ticket07_project_owned_classifier \
  api.tests.test_ticket07_project_owned_dev_evaluator.ProjectOwnedDevelopmentEvaluatorTests.test_procedural_variants_repeat_and_transform_binary_region_support \
  api.tests.test_ticket07_project_owned_dev_evaluator.ProjectOwnedDevelopmentEvaluatorTests.test_texture_only_family_excludes_color_and_brightness_cues \
  -v
```

Result: **8 passed** with system `python3` (Python 3.14, Pillow and NumPy
available). The project API virtual environment lacks Pillow, so the development
evaluator checks could not run there. After producing the report, its
`shortcut_notes.augmentation` description was corrected to mention the bounded
affine warp; the historical JSON report's wording predates that correction, but
its metrics and digest remain unchanged. This check does not establish model acceptance,
real-world material quality, AMD backend compatibility, or RX 7900 GRE resource
fit. Frozen gates and fixture bytes were not changed.
