# Ticket 08 height-detail development v3 latent-normal screen — 2026-10-01

## Decision

The existing per-texel latent-normal estimator is rejected as a material
recovery candidate on the independent height-detail development fixture. It
fits training RGB closely, but its albedo, roughness, and metallic maps remain
far from development truth. It also lacks the caller material-region output
contract, spatial normal consistency, and an exported normal/bump model. It is
diagnostic evidence only; no frozen acceptance data or scorer were used.

## Fixture and isolation

The new seeded fixture has six calibrated views of a plane whose training
observations were rendered with a smooth, spatially varying height-derived
normal field. The height and normals are excluded from candidate inputs. The
candidate loader enforces an exact field allowlist and checks the manifest and
input hashes. Scoring targets are stored separately. The material-map novel
light comparison uses geometric +Z normals, matching the frozen map-only
scorer because the candidate asserts no normal/bump channel.

- Fixture ID: `ticket08-independent-development-v3-height-detail`
- Seed: `20261003`
- Generator SHA-256: `298dcf6832607d72eb6f3ffe9c4c3716c9e1f008995ce125d0ca2c5d5bff7fae`
- Candidate input SHA-256: `d4ae9197232abfbe8cfbe1e78c3a21433c2d5683fe3a456cd36e3ed283ddd591`
- Development target SHA-256: `7d3c0e9053b7fd13897d8dacea43566c63445b8cc3c4208717f79f293140f0a8`
- Estimate SHA-256: `3bf8401bf06986fd2ca0cc53fef85dfa74c99a85e2d8acab5a04714978dce1d9`
- Development score SHA-256: `a4111e0ef3324058de531c3180c87e6133b53558faf0fa81256901af7235dcc6`
- Estimator SHA-256: `d841da1d14773354f387fda7476a84df4616dbe14bac4f638c138c0e16ff0f54`
- Scorer SHA-256: `783a28f2c8c17b3447df9f4f294b7a71ea665a7098c2e38c4d87b446a8911d62`

The estimator opened only the hash-verified candidate input file and the
manifest's training-light definitions. It did not open the target file. The
development-only scorer then opened the separately hash-verified development
target after the estimate was frozen. Held-out data were not present or
accessed. The development fixture test module passed **6/6** in the project API
test venv. The estimate was CPU-only, with no accelerator used; runtime was
49.79 seconds at a 32x32 output resolution, `max_nfev=40`, `min_samples=5`, and
normal prior weight `0.01`.

## Development results

| Metric | Result |
| --- | ---: |
| Base-color linear RGB MAE | 0.15986 |
| Base-color SSIM | 0.26809 |
| Roughness MAE | 0.38785 |
| Metallic MAE | 0.20852 |
| Development novel-light MAE | 0.01143 |
| Visible development coverage | 100% (2,652 texels) |

The training RGB fit MAE was 0.02543 while the material-map errors stayed
large. This confirms that fitting appearance alone can explain observations
without recovering the underlying channels. No numeric acceptance gates were
applied to the development score, and no acceptance threshold changed.

## Next development step

The next candidate must preserve material-region identity and regularize
shading normals spatially or through an integrable height representation while
estimating material maps. It must be developed on this independent fixture and
newly preregistered before each development sweep. A promising synthetic score
will still require the unchanged frozen quality, provenance/rights, adapter,
resource, and RX 7900 GRE gates. Ticket 08 remains acceptance-blocked.
