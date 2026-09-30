# Ticket 08 registered inverse-render repair screen

**Disposition: no new candidate frozen; no scorer invocation.** The registered inverse-render family is not justified for another one-shot dev evaluation using the present allowlisted inputs. The two existing registered candidates are already rejected under the unchanged rubric. A repair that introduces a material-region prior, stronger spatial regularization, or a latent-normal prior would add an unvalidated assumption rather than new independent evidence. This is a source and contract assessment, not a quality or acceptance result.

## Contract reviewed

The frozen rubric is `api/runtime/adapters/pbr/SELECTION.md` (SHA-256 `385c5c2623ded529aac5a92ca2674b86c5ee744948359145a86401fc34d148d7`). The development input scene (`ticket08-three-region-pbr-training-scene-v1.json`, SHA-256 `a460ee742fefb0cd1f29c55bf8f4290de0bbe08556fca61daabb200c28e52dc2`) allowlists only camera-to-world matrices and the fixed three-light training set. The correspondence sidecar manifest (`ticket08-three-region-pbr-correspondence-v1.json`, SHA-256 `02c5a2a01bae87fa5ce34f094e39c6b4ff5224f6ce3873bf700f52c25b30b821`) supplies topology-bound face IDs, barycentrics, visibility, and face UVs. These correspondences establish where each training observation lies on the supplied topology; they do not identify material regions.

The registered input type in `registered_fixed_geometry_v2.py` (SHA-256 `bcb39942b4e62a416aabea9aed50ec03773028627872cd598ede90d9b2a31bd79`) contains mesh positions/UVs/faces, face IDs/barycentrics, registered RGB observations/masks, cameras, and training lights. It has no Ticket 06 segmentation result, region IDs, per-view independently varying lights, or per-light-separated observations. The fixture generator source (`fixture.py`, SHA-256 `d7fc3cc0e581fe889f32c882ce053cc1a10b82a87186919f23ed036d79a84336`) does create region IDs and PBR truth for scoring/render construction; those outputs are excluded from estimator inputs by the allowlist.

## Why a third candidate is not frozen

1. A region-shared BRDF fit is a plausible direction, but there is no truth-isolated region map in this candidate contract. Using the fixture's `material_id`, known region labels, or answer-encoding material names would leak the answer. Adding an unvalidated RGB clustering or spatial-boundary detector would combine segmentation errors and reflectance assumptions without an independent validation set. The existing separate material-region capability is the appropriate source of such evidence once it passes its own acceptance gate.
2. The four calibrated views vary camera direction while reusing the same three directional lights. The registered v2 and v3 fits therefore see repeated observations under a narrow excitation set. Albedo, roughness, metallic, and fine-scale normal-driven shading can trade off. The v3 candidate already adds a latent world-space shading-normal nuisance fit; adding another prior or regularizer after existing results would be parameter tuning without new measured evidence.
3. Raising UV-cell resolution to preserve more texture reduces the number of independent observations per fitted cell; keeping a coarse grid pools different texels and region boundaries. The source does not provide an independently validated choice resolving that tradeoff. A training residual alone cannot decide between explanations because the target material maps and high-frequency normal field are withheld from estimation by design.
4. A dev scorer call can select or reject a frozen candidate, but cannot serve as a tuning signal. No distinct candidate with a predeclared parameterization, training-only stability evidence, and a justified structural prior is ready to freeze. Thus a single dev call now would not be a defensible experiment.

These are feasibility blockers, not proof that all inverse rendering is impossible. Existing rejection evidence remains the quantitative result: `inverse-render-candidate-cpu-rejection-2026-09-25.md` and `fixed-geometry-inverse-render-cpu-rejection-2026-09-25.md`. No score artifact, held-out truth array, or held-out render was opened for this screen.

## Conditions to reopen

- First obtain a separately accepted, truth-isolated Ticket 06 material-region output bound to the same topology revision, with confidence and provenance. It may be supplied as an input; it must not be derived from fixture truth or answer-bearing metadata.
- Declare a materially distinct estimator before dev scoring. Use only training observations/lights/cameras and the registered topology. Any region-shared and texel-varying parameters need an explicit model and unsupported-channel policy.
- On training data only, establish held-in cross-validation and multi-start stability, plus parameter conditioning/uncertainty. These checks should show that the declared fit is identifiable enough to warrant one frozen dev evaluation; a low training residual alone is insufficient.
- Freeze code/inputs/parameters and hashes, then invoke the allowed scorer once. A failed gate rejects that candidate without tuning against dev.
- If maps pass, continue to the RX 7900 GRE and measured <=14 GiB allocated-plus-reserved resource gate. CPU fitting is research evidence only.

The thresholds in `SELECTION.md`, including map, metallic-bias, and novel-light gates, remain unchanged. Ticket 08 remains unaccepted.

## Scope and integrity

This screen did not modify production adapters, fixture assets, scorer code, or selection thresholds. It did not run an estimator or scorer, download assets, or inspect the held-out truth arrays/render. It records why the current source contract does not support a justified new frozen candidate and what independent evidence is required before reopening.
