# Ticket 08 region-spatial v1 development screen — 2026-10-01

## Candidate and decision

A project-owned CPU candidate now fits one PBR parameter set per caller material
region while solving a hidden shading normal at each observed UV cell. The
latent normals are regularized against the face normal and smoothed only across
adjacent texels assigned to the same material region. Conflicting region
assignments remain unknown. The normal nuisance variables are not exported;
only base color, roughness, and metallic are asserted. The candidate preserves a
separate region-map sidecar bound to the exact topology revision.

The vectorized 120-evaluation run is the development lead for one preregistered
frozen-data screen: it has lower base-color, roughness, and SSIM errors than the
300-evaluation run, while both meet the unchanged development reference values.
This is not Ticket 08 acceptance. The sparse optimizer still reached its
function-evaluation cap in both runs (`optimizer_status=0`); the more heavily
iterated result had slightly lower objective but worse recovered maps. No
frozen target metrics were used to select the candidate.

## Development fixture and run identities

The estimator ran on the independent `ticket08-independent-development-v3-height-detail`
fixture. Six calibrated views contain shading from a spatially varying
height-derived normal field. The field is withheld from candidate input and the
input/target files are separate, SHA-bound archives. The runner loaded only the
candidate allowlist. The separate scorer opened development targets after the
estimate and region-map output were frozen. No held-out data or accelerator was
used.

- Candidate-input SHA-256: `d4ae9197232abfbe8cfbe1e78c3a21433c2d5683fe3a456cd36e3ed283ddd591`
- Development-target SHA-256: `7d3c0e9053b7fd13897d8dacea43566c63445b8cc3c4208717f79f293140f0a8`
- Fixture generator SHA-256: `298dcf6832607d72eb6f3ffe9c4c3716c9e1f008995ce125d0ca2c5d5bff7fae`
- Estimator SHA-256: `eb017f6297ba3958e7a08ccf901abf23fb98aafd6a170c6380e7595ee87f35e9`
- Runner SHA-256: `ac66a1ad1d6b827bdf09d34292c331e28d3afacedf9850b07a727a6d0b58171e`
- Scorer SHA-256: `783a28f2c8c17b3447df9f4f294b7a71ea665a7098c2e38c4d87b446a8911d62`
- Selected estimate SHA-256: `65c8ee0bc0fdc9a301c4e64477f2dc0f6fc8af02f6b81bb7c8457b3e1cd44362`
- Selected region-map file SHA-256: `1e50ab4d328656e6d3c7f23a54bb928399c52c7ad056a4700141311f478b914e`
- Topology-bound material-region map digest: `84b6fbf2cbb7828fdadd2c4bff8a806c3a3a24f25e6c5dc8609199ee21bb0d76`
- Selected development score SHA-256: `6ca33e3aec38cdbc35a76355e932375f7897e833b45ee40479efdf83a7feefb3`

The selected estimate used a 32x32 output, 120 maximum evaluations, five-view
minimum, normal spatial weight 0.15, and geometric offset weight 0.015. It took
17.01 seconds on CPU and used 175,504 KiB peak host RSS. The fixture and
region-contract test suite passed **7/7** in the project API test environment.

## Development comparison

| Metric | Vectorized, 120 evaluations | Vectorized, 300 evaluations | Development reference |
| --- | ---: | ---: | ---: |
| Base-color linear RGB MAE | **0.02981** | 0.05122 | <= 0.08 |
| Base-color SSIM | **0.89933** | 0.88172 | >= 0.85 |
| Roughness MAE | **0.02644** | 0.05089 | <= 0.10 |
| Metallic MAE | 0.05603 | **0.05267** | <= 0.10 |
| Development novel-light MAE | **0.00675** | 0.00666 | <= 0.08 |
| Visible coverage | 100% | 100% | informational |
| Optimizer status | max evaluations | max evaluations | must be disclosed |

The non-vectorized 120-step reference took 95.39 seconds and produced the same
estimate bytes as its packaged rerun. The vectorized implementation completes
in 17.01 seconds with slightly different optimizer progress but improves the
base-color map compared with the 300-step result. The 300-step run took 44.91
seconds. All measurements are CPU development evidence, not AMD, frozen, or
production measurements.

## Remaining gates

Before acceptance, this candidate still needs the single frozen PBR quality
comparison under an immutable source/parameter lock, proof that it works with
Modly's actual topology-bound region map and evidence provenance, RX 7900 GRE
execution and memory/latency measurements within the audited budget, integration
into the Structured Asset workflow, and the remaining Ticket 08 criteria.
The prior archive-inventory isolation breach is documented separately; this
screen did not read the frozen archive or scorer. No thresholds were changed.
