# Ticket 08 region-spatial v1 frozen-quality comparison — preregistration

## Frozen candidate

The candidate was selected only on independent development fixture v3. No
frozen target pixel value or score was used to select this configuration. This
screen runs the exact existing Ticket 08 PBR source fixture, training views,
source-derived material-region map, topology-bound correspondence sidecar, and
frozen channel limits. A new output directory and separate one-shot scorer keep
the rejected v2 candidate and its terminal score intact.

The immutable constants are in `region_spatial_v1_frozen_lock.py`:

- Target/fixture archive SHA-256: `b69e9d1853b6c3f207a4f7d737c1e4cbf1542b3c2b7574045901a4e85a7cd7b1`
- Mesh SHA-256: `2d3a8b5ac0bfe44a60c8a7d83ff5b2042a6fac3b715806ac2e454f19b43f52d3`
- Training scene SHA-256: `a460ee742fefb0cd1f29c55bf8f4290de0bbe08556fca61daabb200c28e52dc2`
- Correspondence sidecar SHA-256: `cfbfc146f65451137da58887ef969ec3c06b0fef4c893be29486cfd392933474`
- Material-region map SHA-256: `173ab9622f6df89e1f7ed228216f41467d6487a459ed238a7256aacaf358db6a`
- Estimator SHA-256: `eb017f6297ba3958e7a08ccf901abf23fb98aafd6a170c6380e7595ee87f35e9`
- Resolution: 32x32; maximum function evaluations: 120; minimum samples: 3
- Same-region normal smoothing: 0.15; geometric-normal offset prior: 0.015
- Execution expected by this research implementation: CPU, no accelerator

## Frozen limits

The scorer preserves the existing numeric limits exactly: base-color linear
RGB MAE <=0.08; base-color SSIM >=0.85; roughness MAE <=0.10; metallic MAE
<=0.10; conductor/dielectric metallic bias <=0.08; novel-light linear RGB MAE
<=0.08. Coverage is informational under the existing rubric. No limits are
changed or bypassed.

## Execution and target boundary

The runner may read only `training_observations` and `training_view_masks` from
the combined NPZ, along with the locked GLB geometry, correspondence, training
scene, and source material-region map. It must freeze output maps, an exact
region map, uncalibrated confidence values, and provenance before the scorer
opens any target fields. The scorer validates every frozen identity and output
hash first, accesses only the frozen target allowlist once, and writes one
terminal score report using exclusive file creation.

A prior process accidentally decompressed target/held-out arrays while printing
only archive member names and shapes; this is documented in
`ticket08-normal-detail-domain-gap-2026-10-01.md`. No pixel values were surfaced
or used for candidate tuning. The one-shot score will disclose this history.
This comparison does not measure RX 7900 GRE execution or VRAM and cannot alone
pass the full Ticket 08 AMD/resource or workflow gates.
