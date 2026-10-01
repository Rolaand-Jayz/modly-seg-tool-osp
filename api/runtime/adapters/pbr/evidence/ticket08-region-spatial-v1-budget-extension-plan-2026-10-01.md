# Ticket 08 region-spatial v1 optimizer-budget extension — preregistration

## Fixed comparison

The first region-spatial v1 run reached the declared function-evaluation limit
(`optimizer_status=0`, 45 evaluations) with nonzero optimality. This screen
repeats the exact same candidate, development input bytes, resolution,
regularization, optimizer, and scorer. The only change is increasing the
iteration ceiling from 45 to 120 so the bounded sparse solve has a chance to
converge. This is a convergence check, not a quality-driven parameter sweep.

- Candidate input SHA-256:
  `d4ae9197232abfbe8cfbe1e78c3a21433c2d5683fe3a456cd36e3ed283ddd591`
- Resolution: 32x32
- Max function evaluations: 120 (prior run: 45)
- Minimum samples: 5
- Same-region spatial normal weight: 0.15
- Geometric normal offset weight: 0.015
- All other code and algorithm settings remain fixed.

Freeze the estimate and material-region map before the separate development
scorer opens the v3 target. Compare convergence status, training fit, and all
previously declared development metrics. The development threshold and every
Ticket 08 frozen acceptance limit remain unchanged. This result cannot qualify
an adapter, weights, AMD execution, VRAM, or Ticket 08 acceptance.
