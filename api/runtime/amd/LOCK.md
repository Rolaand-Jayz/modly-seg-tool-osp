# AMD ROCm / Torch-MIGraphX build inputs

This is the source-pinned build record for the RX 7900 GRE (`gfx1100`) candidate
runtime. The base OCI image and direct source repositories are pinned to
immutable identities. It is not a complete transitive package lock. It records
the project-owned build and observed runtime evidence; remaining acceptance
gates are listed in the audited ticket/progress record.

| Input | Identity |
| --- | --- |
| AMD PyTorch base image | `docker.io/rocm/pytorch@sha256:a223aee17aef5d21c3b9f63436dd19d27d1c665ec8b2f40011c9546cabae2a80` |
| ROCm devel distribution | `rocm[devel]==7.14.0` from AMD's multi-arch wheel index |
| MIGraphX | `ROCm/AMDMIGraphX@4bcfe75b225e4e0b9387fe14645cb1c7216e6742` |
| rbuild | `ROCm/rbuild@6b12f6a10c85a6fc2c0b906da6478d92b4957e29` |
| Torch-MIGraphX | `ROCm/torch_migraphx@e551a861cf8fc0865d81920fa6f53db210763eed` |
| GPU target | `gfx1100` |
| Python ABI | CPython 3.12, inherited from base image |
| NumPy | `1.26.4` |
| tabulate | `0.9.0` |

## Observed build and package inventory

The project-local OCI image built successfully from the `Containerfile` with
image ID `c96b56796c753d749cb02b31b88da7567c3712861d214bfe26452d21bd7e6b8d`
and local Podman manifest digest
`sha256:64fb89f2167f56a7381529e8e4fb69b322c3685f9805ea301263477c25aa0554`.
The AMD devel wheel installed as `rocm-sdk-devel==7.14.0`. The Torch-MIGraphX
wheel built in the image is `torch_migraphx==1.2`, wheel SHA-256
`b92599de5a88f89f612663760f9b0fbb5dec78177a6c063afca63ef4c153e11d`.

The captured 126-distribution Python inventory is
[`package-inventory.json`](package-inventory.json); its canonical inventory
content digest is
`sha256:c637b396df8593357bcc7a986bc60ad2c19dba46999202379d5df5e9a4276ebf`,
and the checked-in inventory file SHA-256 is
`e42e1015084cc994fdebd83c94afae62f60364b6334fc5b001bdb0cbba0922cd`.
The installed dpkg package inventory has 360 nonempty `name=version` records;
its SHA-256 is
`33de241f7e13a753782214e2498d2d815ac5bf15b284c7534412af8d29cd39ae`.
The package list and compiler versions are recorded in
[`native-package-inventory.txt`](native-package-inventory.txt) and
[`native-toolchain-versions.txt`](native-toolchain-versions.txt). Their SHA-256
values are `33de241f7e13a753782214e2498d2d815ac5bf15b284c7534412af8d29cd39ae`
and `2095d1c69a9236552c352600000522c10b525f8bef408e773022701f70719ca7`.
Observed native versions: ROCm SDK 7.14.0, CMake 3.28.3, AMD clang 23.0.0git,
Git 2.43.0, PyTorch 2.11.0+rocm7.14.0, HIP 7.14.60850, Python 3.12.3,
MIGraphX `2.16.0.dev+20250912-17-575-g4bcfe75b2`, Torch-MIGraphX 1.2.
[`evidence/native-library-evidence.json`](evidence/native-library-evidence.json)
records SHA-256 hashes of the installed MIGraphX C, Python, core, GPU, device,
reference, ONNX, and TensorFlow shared libraries. Its checked-in file hash is
`e784eeb7674051b90a6ed84892e7c760ae2c9181c5521077982d94298a113d45`. It also
records `torch.version.cuda: null`, HIP `7.14.60850`, and no `libcuda` or
`libnvidia` sonames in `ldd` results for those native libraries. The linked
library listing and explicit check are in
[`evidence/migraphx-ldd.txt`](evidence/migraphx-ldd.txt) and
[`evidence/no-nvidia-check.txt`](evidence/no-nvidia-check.txt).

## GPU probe result

The final [`evidence/migraphx-probe.json`](evidence/migraphx-probe.json) passed
with exit code 0. Its FP16 3-layer dense fixture (batch 1024, 2048 inputs,
4096-wide hidden layers, 1024 outputs) ran through the Modly runtime's
`torch.compile(backend="migraphx")` integration on RX 7900 GRE/gfx1100. Across
31 timed repetitions after warm-up, PyTorch ROCm took 3.7041 ms and MIGraphX
took 1.1811 ms (3.14x). Modly's eager-vs-candidate comparison passed at
`atol=rtol=0.005`; the report includes deterministic input/weight/output
digests and all backend identities.

The same run separately compiled an FP32 candidate with an intentionally
strict finite `min_speedup=1,000,000`. Modly rejected it as `rejected_slow`,
selected PyTorch ROCm for the whole named module, and matched the independent
eager output at `atol=0.0002`, `rtol=0.002`. This verifies the explicit fallback
seam without assuming graph partitioning.

The final [`evidence/sequential-memory-probe.json`](evidence/sequential-memory-probe.json)
also passed with exit code 0. Two sequential 1024×2048 FP16 inference regions
both selected MIGraphX and completed. After each compiled/output/module
reference was released via `release_region`, allocated/reserved VRAM returned
to 33,554,432/33,554,432 bytes; both stages reported a 232,820,736-byte peak.
The initial process allocator state was 0/0 bytes. The measured post-release
plateau was identical after both runs. Run IDs and telemetry are in the JSON
evidence.

The independently seeded repeated workload results are in
[`evidence/fp16-repeated-benchmark.json`](evidence/fp16-repeated-benchmark.json)
(SHA-256 `df590ac42d64bc9ac0d5ad355384508decc460ba282db1625ae64dff170907d5`).
All three 31-repetition pairs selected MIGraphX: eager ROCm ranged 3.6856–
3.7220 ms and MIGraphX ranged 1.1783–1.1944 ms. Each candidate compiled and
passed the same stated numerical tolerance.

MIGraphX's pinned `requirements.txt` declares these additional direct build
inputs: abseil-cpp `20250512.0`, protobuf `v30.0`, nlohmann/json `v3.8.0`,
pybind11 `3e9dfa2866941655c56877882565e7577de6fc7b`, msgpack-c `cpp-3.3.0`,
SQLite `3.50.4`, rocm-cmake
`1d4652ae2ec0e44a67a7c415dd7e51c88a6aa68d`, composable_kernel
`ad0db05b040bacda751c65c705261b8a0a7ed25d`, Eigen `5.0.1`, and rocMLIR
`518955cab3ec2a53cbb03216cb55ac6ab47ee1da`. rbuild obtains its ROCm build
recipes through the dependency declaration in the pinned MIGraphX source.

The wheel dependency closure, AMD SDK wheel hashes, OS package repository
snapshot, build tool transitive dependencies, and fetched archive hashes are
not locked here. The MIGraphX source's rbuild recipe dependency
`ROCm/rocm-recipes` was downloaded as `archive/HEAD.tar.gz` during this build;
the dependency recipes themselves were not pinned to a content identity in the
executed image build. Thus the recorded image and inventories identify what ran,
but the recipe is not byte-for-byte reproducible yet. Do not infer full Ticket
02 acceptance from a successful native build alone.
