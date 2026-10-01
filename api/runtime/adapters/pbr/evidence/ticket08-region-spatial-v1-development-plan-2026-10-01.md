# Ticket 08 region-spatial v1 development plan — 2026-10-01

## Candidate fixed before scoring

This CPU development screen tests whether sharing base-color/roughness/metallic
parameters across caller material regions, while fitting a hidden shading normal
per UV cell and smoothing it only across neighbors within the same region,
reduces the appearance/material ambiguity seen in v3. The hidden normals are
nuisance variables and will not be emitted as a normal/bump map. A conflicting
UV texel is left unknown instead of mixing region identities.

- Fixture: `ticket08-independent-development-v3-height-detail`
- Candidate inputs: SHA-256
  `d4ae9197232abfbe8cfbe1e78c3a21433c2d5683fe3a456cd36e3ed283ddd591`
- Resolution: 32x32
- Maximum nonlinear fit evaluations: 45
- Minimum multiview samples per texel: 5
- Same-region spatial normal weight: 0.15
- Geometric normal offset weight: 0.015
- Optimizer: bounded sparse SciPy TRF with LSMR, `soft_l1`, `f_scale=0.025`
- Regions share one PBR triplet per region; only observed texels are written.

The estimator receives only the candidate-input allowlist. All maps will be
written once before the separate development scorer opens its target file. No
parameter sweep is preregistered in this plan. The candidate will be rejected
if the development maps do not substantially improve over the previous v3
latent-normal result (base-color MAE 0.15986, SSIM 0.26809, roughness MAE
0.38785, metallic MAE 0.20852) while preserving visible coverage and exact
region assignment. This is a development decision rule, not a frozen acceptance
gate. Regardless of outcome, no frozen score, acceptance threshold, model
weight, or RX 7900 GRE run is part of this screen.
