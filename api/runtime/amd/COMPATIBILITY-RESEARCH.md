# AMD adapter compatibility research

**Status: no accepted immutable adapter lock.** This record evaluates whether
the currently identified AMD PyTorch image can anchor a project-owned
Torch-MIGraphX runtime. It does not claim runtime compatibility or hardware
acceptance.

## Candidate image under review

The exact image used by AMD's current `ROCm/torch_migraphx` Dockerfile has
been resolved through Docker Hub to this immutable digest:

```text
rocm/pytorch:rocm7.14_ubuntu24.04_py3.12_pytorch_release_2.11.0
sha256:a223aee17aef5d21c3b9f63436dd19d27d1c665ec8b2f40011c9546cabae2a80
```

This matches the upstream recipe's tag. The digest pins Ubuntu 24.04,
Python 3.12, PyTorch 2.11.0, and the ROCm 7.14 runtime image. A base-image
digest alone does not pin adapter build inputs added later.

## Upstream recipe findings

The official [ROCm/torch_migraphx README](https://github.com/ROCm/torch_migraphx/)
currently describes its default recipe as PyTorch 2.11, ROCm 7.14, and Python
3.12 with a matching MIGraphX release. Its
[Dockerfile](https://github.com/ROCm/torch_migraphx/blob/master/Dockerfile)
uses `rocm/pytorch:rocm7.14_ubuntu24.04_py3.12_pytorch_release_2.11.0`, installs `rocm[devel]`
using the version returned by the image's `rocm-sdk`, clones MIGraphX at the
`rocm-7.14` ref, and downloads rbuild from
`https://github.com/RadeonOpenCompute/rbuild/archive/master.tar.gz`.

Some source identities can be made immutable in a future build recipe:

* AMD's signed MIGraphX `rocm-7.14` release resolves to commit
  [`4bcfe75b225e4e0b9387fe14645cb1c7216e6742`](https://github.com/ROCm/AMDMIGraphX/commit/4bcfe75b225e4e0b9387fe14645cb1c7216e6742).
  This is an immutable source identity, but it does not pin the other build
  inputs or establish `gfx1100` qualification.
* The inspected upstream Torch-MIGraphX build recipe resolves to commit
  [`e551a861cf8fc0865d81920fa6f53db210763eed`](https://github.com/ROCm/torch_migraphx/commit/e551a861cf8fc0865d81920fa6f53db210763eed).
* The upstream `rbuild` repository resolves to commit
  [`6b12f6a10c85a6fc2c0b906da6478d92b4957e29`](https://github.com/ROCm/rbuild/commit/6b12f6a10c85a6fc2c0b906da6478d92b4957e29).
* PyPI's [torch-migraphx 1.1 provenance](https://pypi.org/project/torch-migraphx/)
  identifies source commit
  [`f28c35da5a0e268ce8cc0234694b3a4c3ad352bd`](https://github.com/ROCm/torch_migraphx/commit/f28c35da5a0e268ce8cc0234694b3a4c3ad352bd)
  and source archive SHA-256
  `920cb700b05d09aee0f97cb6909a95f224a990b7b8c39eca778638cefa2a6129`.
  The published wheel hash is
  `2eec69d23dbe2a81ddbffc3204fd6446f65859657c83ac74d8e1414dc2e397a7`.
  These identify a release artifact; the available evidence does not establish
  that this exact release was built and qualified against the candidate image
  and MIGraphX commit.

## Why no project lock or Dockerfile was added

The available primary-source evidence does not describe one complete,
reproducible source/package set for the candidate image:

1. The upstream Dockerfile resolves the image tag and uses mutable source
   references. The base image and all three source repositories are now
   resolved to exact content identities, and the project build can use those
   immutable references.
2. It resolves `rocm[devel]` dynamically from the SDK version in the image,
   through the AMD wheel index. The recipe does not supply hashes for that
   package or the complete resolved dependency closure.
3. AMD's [ROCm transition guide](https://rocm.docs.amd.com/en/docs-7.14.1/about/transition-guide-TheRock.html)
   says MIGraphX moved to ROCm-Extras, separate from the ROCm Core SDK. The
   Torch-MIGraphX recipe builds MIGraphX from source, but the inspected sources
   do not establish ABI/runtime compatibility between that build, the
   candidate TheRock-based PyTorch image, and the `gfx1100` target.
4. The upstream README calls its combination a matching release, but no
   inspected AMD source gives a complete, digest-pinned qualification tuple for
   this exact base-image digest and `gfx1100`. A successful build or numerical
   and performance gate on the target GPU is still required by Ticket 02.

These identities allow a project-owned immutable build recipe. They do not
yet constitute a complete lock: the ROCm SDK dependency closure, apt build
inputs, and Python build requirements still need exact version/hash capture,
and actual `gfx1100` qualification remains mandatory. Ticket 02 still requires
real correctness, usefulness, fallback, telemetry, and cleanup evidence before
the runtime can be accepted.

## Evidence needed to close the compatibility question

1. Complete AMD/upstream-confirmed compatibility evidence for the exact base
   image digest, source commits, and `gfx1100`.
2. Resolve and record full immutable identities and hashes for the ROCm SDK
   package closure, Python build packages, apt inputs, and native build inputs.
3. Build the project-owned adapter from that lock, then exercise its imports and
   actual MIGraphX compilation/inference on the RX 7900 GRE (`gfx1100`).
4. Keep the result separate from base-image metadata: record numerical
   agreement against ROCm PyTorch, measured speed and memory, fallback
   behavior, and release of allocations between sequential model loads.

Until those inputs and results exist, the exact compatibility question remains
open; no implementation or acceptance status is inferred from source
inspection alone.
