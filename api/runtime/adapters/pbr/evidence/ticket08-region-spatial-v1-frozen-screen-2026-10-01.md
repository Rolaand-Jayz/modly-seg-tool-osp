# Ticket 08 region-spatial v1 frozen screen — 2026-10-01

## Terminal result

The separately locked region-spatial v1 candidate was estimated from the exact
frozen training allowlist, source-derived material-region map, and topology-bound
correspondence. A one-shot scorer then opened only its declared frozen target
fields. The candidate is **rejected**: linear base-color MAE is 0.08604, above
the unchanged <=0.08 limit. This failure alone means the candidate cannot pass,
regardless of the metallic-bias audit below. No follow-up score, target read, or
candidate tuning occurred.

| Metric | Result | Existing limit | Screen result |
| --- | ---: | ---: | --- |
| Base-color linear RGB MAE | 0.08604 | <= 0.08 | Fail |
| Base-color SSIM | 0.87240 | >= 0.85 | Pass |
| Roughness MAE | 0.02275 | <= 0.10 | Pass |
| Metallic MAE | 0.02419 | <= 0.10 | Pass |
| Reported conductor/dielectric value | 0.95385 | <= 0.08 | See metric defect |
| Novel-light linear RGB MAE | 0.03102 | <= 0.08 | Pass |
| Visible coverage | 93.33% (53,760/57,600) | Informational in rubric | Reported |

Coverage is reported, not used as a separate numeric gate, consistent with the
existing selection rubric. Channel scores in this terminal report were
calculated over `visible_mask & candidate_observed`; therefore they cover only
53,760 texels. The audited rubric says to score fixed visible texels, so
handling of unobserved visible texels needs an explicit, non-weakening metric
interpretation before any future acceptance decision. This candidate already
fails base-color MAE on the reported subset.

## Frozen identities

- Candidate: `modly.project-owned-region-spatial-inverse-v1`
- Fixture SHA-256: `b69e9d1853b6c3f207a4f7d737c1e4cbf1542b3c2b7574045901a4e85a7cd7b1`
- Source material-region map SHA-256: `173ab9622f6df89e1f7ed228216f41467d6487a459ed238a7256aacaf358db6a`
- Candidate estimate SHA-256: `386bddf4b2edc4f5e24736bb6ebdead3f04127414b6f902142ea9ba7742413e0`
- Topology-bound output region map SHA-256: `04c4b7e88c382930fa8553235450e0d057475dfc5df6d424d6e9d4e07ed3ecc1`
- Training-run report SHA-256: `3dd5bfbf5a379a8d070151be9fa6e38524c29441fd830988933afd54b7136470`
- Terminal score report SHA-256: `dc27a7d5ef0288fff04cba6b3add0521211d7bad899baeb04c17e8cfaa3dde90`
- Estimator SHA-256: `eb017f6297ba3958e7a08ccf901abf23fb98aafd6a170c6380e7595ee87f35e9`
- Frozen lock SHA-256: `42175f89eb16b537b2e439d232012a3b67fea554cd873eb526a5f6f40401692d`
- Runner SHA-256: `c378ccf4615fbfca366a3c9bbaa7a6c9be7a6c45ec378baa8f83de3a1336cae4`
- Scorer SHA-256: `25ca695d4ee11cb6964e944fc3a748f2ac3db46047630e6557aed1e5f6392be8`

The training estimate used 32x32 maps, 120 maximum evaluations, and a minimum
of three source-view samples per cell. It completed in 52.30 seconds on CPU,
with 413,696 KiB peak host RSS and zero accelerator devices. The optimizer hit
its iteration limit. The scorer ran once on CPU in 0.056 seconds, with 165,624
KiB peak host RSS. This does not establish RX 7900 GRE runtime or VRAM gates.

## Scorer defect found after the one-shot result

`score_metallic_bias` currently returns the absolute difference between the
candidate's mean conductor metallic value and its mean dielectric metallic
value. It does not compare that estimated separation with the corresponding
separation in the reference map. The audited rubric calls for conductor/
dielectric *bias*, which requires comparing predicted and reference group
separations. Therefore the reported 0.95385 is not a valid bias measurement.
The output and terminal report are preserved unchanged; this metric was not
recomputed from frozen targets. A corrected reference-relative metric must be
used for any future candidate, with its own locked one-shot protocol. The
candidate remains rejected because its base-color MAE already fails.

The scorer also records the earlier archive-inventory incident: target/held-out
arrays were once decompressed while printing only member names and shapes; no
pixel values were surfaced or used for candidate tuning. This historical
access remains disclosed and no further frozen scoring is authorized against
this candidate output.
