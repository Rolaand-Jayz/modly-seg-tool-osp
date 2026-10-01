# Ticket 12 authored golden fixture package

Versioned CPU-side reference index for acceptance fixture construction. The
package reuses the procedural car development mesh by digest and carries a
small self-contained NPZ with the synthetic three-region PBR fixture's authored
truth maps, region IDs, and plane geometry. The local PBR archive excludes
rendered observations and runtime/model outputs. It adds explicit
unknown/ambiguous contract oracles and a deterministic topology invalidation
case. It contains no neural predictions, SUV content, weights, or acceptance
run results.

`manifest.json` records each source, its provenance, the defensible oracle, and
cases that remain omitted because no checked truth asset currently exists.
Digest changes fail the integrity test until the fixture review updates the
manifest intentionally. These fixtures do not qualify either adapter or meet
Ticket 12's hardware/end-to-end gates by themselves.

Run the focused integrity check from the repository root:

```sh
PYTHONPATH=api python -m unittest api.tests.test_ticket12_golden_fixtures -v
```
