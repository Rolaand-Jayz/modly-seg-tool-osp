# Ticket 08 normal-detail development gap — 2026-10-01

## Finding

The recent region-inverse v2 development result does not exercise the failure
mode present in the frozen PBR scene. Development fixture v2 builds a planar
mesh and its generated observations use a constant `(0, 0, 1)` normal field.
The frozen candidate's recorded inputs describe a six-triangle planar mesh,
while the frozen scene contract includes height-derived surface detail. The
frozen v2 estimate reported large per-region fit errors for two regions and
failed base-color MAE/SSIM, roughness, metallic, and conductor/dielectric bias.
Its development-only lambda sweep therefore does not justify further tuning
the base-color region prior.

The earlier registered latent-normal candidate is not a solution: its one-shot
score also failed base-color MAE/SSIM, roughness, metallic MAE, and metallic
bias. It emitted no normal or bump assertion, and its per-texel shading-normal
fit has no spatial or height-integrability model. No candidate passed Ticket
08, and no threshold or frozen artifact was changed.

## Evidence boundary

This diagnosis uses only the committed development fixture generator, frozen
runner metadata, and existing terminal score reports. Frozen target pixels,
material maps, and held-out reference were not read or scored in this
investigation. In particular, the terminal frozen v2 scorer was not rerun.
The runtime archive inventory showed that training and scoring members share a
container, so future analysis must load only explicitly allowlisted training
members rather than iterating the complete archive.

Relevant identities and reports:

- Development v2 generator: `api/runtime/adapters/pbr/development_fixture_v2.py`
- Frozen v2 training report: `.modly-amd-runtime/ticket08-source-bound-map-20261001/ticket08-region-inverse-v2-frozen-run.json`
- Frozen v2 terminal score: `api/runtime/adapters/pbr/evidence/ticket08-region-inverse-v2-source-bound-score-2026-10-01.md`
- Development lambda sweep: `api/runtime/adapters/pbr/evidence/ticket08-region-inverse-v2-albedo-prior-sweep-results-2026-10-01.md`
- Latent-normal rejection: `api/runtime/adapters/pbr/evidence/ticket08-registered-latent-normal-v3-quality-score.json`

## Next development action

Before another estimator change, add a separate, seeded, truth-isolated
development fixture whose training observations include spatially varying
surface normals from a documented height field. Keep the candidate input and
scoring target files separate and hash-bound. Develop a spatially coherent
normal/detail nuisance model on that fixture; only proceed to a new frozen
candidate if it improves material-map recovery without asserting unsupported
normal/bump channels. Any later frozen evaluation must use a newly preregistered
candidate and preserve the existing one-shot scorer and all acceptance limits.

This evidence changes the development plan only. Ticket 08 remains
acceptance-blocked, and AMD/RX 7900 GRE resource and runtime gates remain
unverified for this candidate.
