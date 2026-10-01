# Shared GPU run policy

Modly may run GPU work while the desktop and other applications are active, as
authorized by the project owner. Project-owned PyTorch GPU inference adapters
must sample both PyTorch's free memory and system-wide VRAM use before model
load, then apply `api/runtime/amd/gpu_budget.py`. The budget uses the lower
free-memory estimate, leaves 4 GiB for the desktop and other applications,
caps one adapter at 14 GiB, and refuses to start with less than 2 GiB available
after the reserve. If system-wide usage cannot be matched uniquely to the
active GPU, the adapter refuses to start.

The limit applies only to PyTorch's caching allocator. It does not reserve
physical GPU memory or limit non-PyTorch allocations and does not prevent GPU
compute contention. GeoSAM2 runs must use
`scripts/monitored_geosam2_workflow.py`, which samples host VRAM every two
seconds and, at the reserve, targets that run's container ID and delegated
cgroup. A launcher process signal or Podman stopped-state response alone is
not proof that the inference process has exited; confirm the cgroup is empty
and KFD has no run process. The run must write only to a new, project-owned
output path. Model/cache/runtime files are read-only inputs unless a specific
stage contract names an output.

This policy is resource containment, not proof that concurrent desktop
rendering is unaffected. Ticket 12 still requires measured per-stage VRAM,
latency, backend, and successful end-to-end workflow evidence on the RX 7900 GRE.
