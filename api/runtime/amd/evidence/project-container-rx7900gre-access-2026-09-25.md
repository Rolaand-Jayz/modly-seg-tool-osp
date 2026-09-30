# Project container RX 7900 GRE access — 2026-09-25

## Result

The existing project-owned container image
`localhost/modly-amd-migraphx:ticket02` runs with the RX 7900 GRE visible when
Podman is invoked through the managed host command path and given a
project-owned `XDG_RUNTIME_DIR`. No network was enabled and no host package or
custom Python installation was modified.

Image ID inspected from the project storage: `c96b56796c753d749cb02b31b88da7567c3712861d214bfe26452d21bd7e6b8d`.

Runtime result:

- PyTorch: `2.11.0+rocm7.14.0`
- HIP: `7.14.60850`
- `torch.cuda.is_available()`: `True` (PyTorch ROCm API)
- Device count: `2`
- Device 0: `AMD Radeon RX 7900 GRE`

The restricted shell did not expose `/dev/kfd` or `/dev/dri`; therefore its
ROCm probe saw only the CPU. The managed host launch below exposes devices to
the project container and is required for subsequent hardware probes in this
execution environment.

## Exact read-only runtime command

```bash
XDG_RUNTIME_DIR="$PWD/.modly-amd-runtime/xdg-runtime" \
  podman --root "$PWD/.modly-amd-runtime/storage" \
  --runroot "$PWD/.modly-amd-runtime/run" \
  run --rm --userns=host --network=none \
  --device /dev/kfd --device /dev/dri \
  --entrypoint python localhost/modly-amd-migraphx:ticket02 \
  -c 'import torch; print("torch",torch.__version__,"hip",torch.version.hip,"cuda",torch.cuda.is_available()); print("device_count",torch.cuda.device_count()); print("device0",torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")'
```

The output was:

```text
torch 2.11.0+rocm7.14.0 hip 7.14.60850 cuda True
device_count 2
device0 AMD Radeon RX 7900 GRE
```

This confirms device access only. It does not establish any ticket's model
quality, MIGraphX/ROCm parity, resource use, latency, or end-to-end acceptance.
