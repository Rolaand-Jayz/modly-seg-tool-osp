# Ticket 08 synthetic light-excitation conditioning probe (2026-09-27)

## Scope

This small CPU experiment checks one structural question raised by the rejected
inverse-render fits: whether repeated camera views under the same lights make
the fitted albedo, roughness, and metallic parameters locally hard to
distinguish. It uses only a hand-written synthetic planar sample and the
existing GGX forward function. It does not load the PBR fixture, fit a map,
compare estimates against known material values, invoke a scorer, or inspect
truth.

The probe differentiates the existing `_ggx_batch` renderer at one arbitrary
interior parameter point using central finite differences. It compares four
camera directions under either (a) the same three calibrated lights for every
view, or (b) a rotated three-light rig per view. It reports the Jacobian
singular values, numerical rank, and condition number only. This is a local
forward-model sensitivity check, not an estimator-quality measurement.

## Result

Both synthetic Jacobians have rank 5. Repeating the same light rig gives a
condition number of **19.19**; rotating the rig for each view gives **12.27**
(about 1.56 times lower). The smallest singular value increases from 0.02251
to 0.03575. Under this synthetic setup, varied illumination provides somewhat
stronger local separation among the five fitted parameters, but it does not
establish that any fitted map would meet a quality gate.

The current registered Ticket 08 input contract exposes a single shared
training-light set, not independently varying per-view rigs. Therefore the
second scenario is not available to the current candidate without an additive
per-view calibrated-light input seam. The result does not justify modifying
the rejected estimator or invoking the development scorer. It identifies a
potentially useful acquisition/interface requirement only if genuine source
observations later provide calibrated, varied illumination.

## Reproduction and identity

Command:

```sh
PYTHONPATH=.modly-amd-runtime/ticket08-scientific-overlay:api \
  .modly-amd-runtime/api-test-venv/bin/python \
  api/runtime/adapters/pbr/research/ticket08_light_excitation_conditioning.py
```

The project Python 3.12 environment needs its existing SciPy overlay because
the imported research module imports SciPy at module load, although this probe
calls only its NumPy GGX forward function. No package was installed or changed.

- Probe SHA-256: `d21e581be1251497ed1f6730ae71b1430abcddde9d2e51dce7b0b992fe937266`
- Existing forward-model module SHA-256: `46d6973845dd7dae1e704d466d63abf24948b1437ecbc8fc3d39aec6517e6d0c`
- Result: successful CPU execution; fixture/truth access and estimator/scorer invocation are explicitly false in the JSON output.

Ticket 08 remains blocked. Frozen gates and prior candidate rejections are
unchanged.
