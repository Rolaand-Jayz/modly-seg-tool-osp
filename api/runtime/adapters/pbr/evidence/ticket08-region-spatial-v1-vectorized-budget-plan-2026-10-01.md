# Ticket 08 region-spatial v1 vectorized convergence check — preregistration

The same v1 candidate, v3 training archive, 32x32 output grid, regularization,
and scorer will be used. Only the nonlinear function-evaluation ceiling changes
from 120 to 300 because the 120-step vectorized run still reports the maximum
iteration status. This tests convergence after removing repeated per-cell
Python overhead; it is not a target-quality-driven parameter search.

- Candidate input SHA-256:
  `d4ae9197232abfbe8cfbe1e78c3a21433c2d5683fe3a456cd36e3ed283ddd591`
- Resolution: 32x32
- Maximum function evaluations: 300 (prior vectorized run: 120)
- Minimum samples: 5
- Same-region spatial normal weight: 0.15
- Geometric normal offset weight: 0.015
- Same sparse TRF/LSMR solver, robust loss, and data.

The estimate and exact topology-bound region-map sidecar will be frozen before
the separate v3 development target is scored. Acceptance thresholds are not
applied or changed. This run cannot qualify frozen quality, an AMD backend,
VRAM, latency on RX 7900 GRE, or a production adapter.
