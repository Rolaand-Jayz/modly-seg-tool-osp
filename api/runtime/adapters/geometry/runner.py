"""Pinned Hunyuan3D Mini Turbo geometry adapter for Modly's AMD runtime."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

from runtime.adapters.geometry.hunyuan_source import load_shape_pipeline
from runtime.adapters.geometry.stage import GeometryGenerationResult
from runtime.amd.gpu_budget import apply_gpu_budget


HUNYUAN_SOURCE_REVISION = "f8db63096c8282cb27354314d896feba5ba6ff8a"
HUNYUAN_WEIGHTS_REVISION = "f90a0f7df7d5e6f71109cf333f6a95a0ae3194a6"
HUNYUAN_WEIGHTS_REPO = "tencent/Hunyuan3D-2mini"
DIT_SHA256 = "bdbcef30dd0149a281e17d5b5b1fdad1122c904e098a42f3100e04e03c247bc4"
VAE_SHA256 = "5dcaca67a8da9e7079fb7b55714a572d0651ac95983bb43099913feaf6738f94"
SOURCE_ARCHIVE_SHA256 = "52b7c9647e457f9b4055e3488e06e675117eae01d6c48b28d8ec1b869a697b2c"
DIT_SUBFOLDER = "hunyuan3d-dit-v2-mini-turbo"
VAE_SUBFOLDER = "hunyuan3d-vae-v2-mini-turbo"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _candidate_root() -> Path:
    configured = os.environ.get("MODLY_GEOMETRY_MODEL_ROOT")
    if configured:
        return Path(configured).resolve()
    container_root = Path("/models/hunyuan-mini-turbo")
    if container_root.is_dir():
        return container_root
    project_root = Path(__file__).resolve().parents[4]
    return project_root / ".modly-amd-runtime" / "models" / "hunyuan-mini-turbo"


def _ensure_project_migraphx_python_path(rocm_python_path: Path = Path("/opt/rocm/lib")) -> None:
    """Restore the pinned image's MIGraphX module path inside this worker."""
    if rocm_python_path.is_dir() and str(rocm_python_path) not in sys.path:
        sys.path.append(str(rocm_python_path))


def _preserve_pipeline_model_controls(source_model: Any, routed_model: Any) -> None:
    """Keep the upstream pipeline's capability flags on the routed module."""
    for name in ("guidance_embed", "guidance_cond_proj_dim"):
        if hasattr(source_model, name):
            setattr(routed_model, name, getattr(source_model, name))


def _enable_hunyuan_model_cpu_offload(pipeline: Any, torch: Any) -> None:
    """Route Hunyuan's three real model components through its upstream hook sequence.

    Hunyuan's custom pipeline implements Diffusers' offload method but does not
    expose DiffusionPipeline.components. Its pipeline call also reads
    ``self.device`` directly, so preserve CUDA as the input/execution device
    after the official hook installer has moved module weights to CPU.
    """
    pipeline.components = {
        "conditioner": pipeline.conditioner,
        "model": pipeline.model,
        "vae": pipeline.vae,
    }
    pipeline.enable_model_cpu_offload(device="cuda")
    pipeline.device = torch.device("cuda")


def _add_candidate_dependencies(root: Path) -> Path:
    package_root = root / "python-packages"
    source_root = root / "source"
    if not package_root.is_dir() or not source_root.is_dir():
        raise RuntimeError("Pinned Hunyuan source and candidate-local dependencies must be staged in the project runtime")
    for package in (str(package_root), str(source_root)):
        if package in sys.path:
            sys.path.remove(package)
        sys.path.insert(0, package)
    # The headless process runner intentionally replaces inherited PYTHONPATH
    # with an explicit extension environment. The pinned project image installs
    # MIGraphX's Python module in this image-local path, so restore it inside
    # the extension worker before AMDInferenceRuntime attempts its import.
    _ensure_project_migraphx_python_path()
    os.environ["HY3DGEN_MODELS"] = str(root / "weights")
    return source_root


def _verify_candidate(root: Path) -> tuple[Path, str]:
    source_root = _add_candidate_dependencies(root)
    source_archive = root / "source.tar.gz"
    if not source_archive.is_file() or _sha256(source_archive) != SOURCE_ARCHIVE_SHA256:
        raise RuntimeError("Pinned Hunyuan source archive is missing or has a SHA-256 mismatch")
    dit_path = root / "weights" / DIT_SUBFOLDER / "model.fp16.safetensors"
    vae_path = root / "weights" / VAE_SUBFOLDER / "model.fp16.safetensors"
    for path, expected in ((dit_path, DIT_SHA256), (vae_path, VAE_SHA256)):
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"Pinned Hunyuan model weight is missing or has a SHA-256 mismatch: {path.name}")
    combined = hashlib.sha256(f"dit:{DIT_SHA256}\nvae:{VAE_SHA256}\n".encode("ascii")).hexdigest()
    return source_root, f"sha256:{combined}"


class HunyuanMiniTurboAdapter:
    """Image-conditioned shape generation; texture/PBR estimation stays downstream."""

    model_id = "tencent.Hunyuan3D-2mini.shape"

    def __init__(self) -> None:
        import torch

        if not getattr(torch.version, "hip", None) or not torch.cuda.is_available():
            raise RuntimeError("Hunyuan generation requires the project ROCm container and an available AMD HIP device")
        self.torch = torch
        self.gpu_budget = apply_gpu_budget(torch, torch.cuda.current_device())
        adapter_started = time.perf_counter()
        self.root = _candidate_root()
        source_root, self.weights_digest = _verify_candidate(self.root)
        pipeline_class = load_shape_pipeline(source_root)
        self.cpu_offload_enabled = os.environ.get("MODLY_GEOMETRY_HUNYUAN_CPU_OFFLOAD") == "1"
        self.runtime = self._runtime_type()(detail_log=Path(os.environ.get(
            "MODLY_AMD_RUNTIME_LOG", str(self.root / "runtime" / "amd-runtime.jsonl")
        )))
        self.load_started = time.perf_counter()
        self.pipeline = pipeline_class.from_pretrained(
            ".",
            device="cpu" if self.cpu_offload_enabled else "cuda",
            dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
            subfolder=DIT_SUBFOLDER,
        )
        self.model_load_ms = (time.perf_counter() - self.load_started) * 1000.0
        self.module_report: dict[str, object]
        self._route_denoiser()
        if self.cpu_offload_enabled:
            _enable_hunyuan_model_cpu_offload(self.pipeline, self.torch)
        self.adapter_setup_ms = (time.perf_counter() - adapter_started) * 1000.0
        self._write_probe_state(
            model_loaded=True,
            phase="ready_for_generation",
            model_load_ms=self.model_load_ms,
            adapter_setup_ms=self.adapter_setup_ms,
            cpu_offload_enabled=self.cpu_offload_enabled,
            backend=self.module_report.get("backend"),
            weights_digest=self.weights_digest,
        )

    def _write_probe_state(self, **updates: object) -> None:
        state_path = os.environ.get("MODLY_GEOMETRY_PROBE_STATE_FILE")
        if not state_path:
            return
        path = Path(state_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        previous: dict[str, object] = {}
        if path.is_file():
            try:
                previous = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                previous = {}
        previous.update(updates)
        path.write_text(json.dumps(previous, sort_keys=True) + "\n", encoding="utf-8")

    @staticmethod
    def _runtime_type():
        from services.amd_runtime import AMDInferenceRuntime
        return AMDInferenceRuntime

    def _route_denoiser(self) -> None:
        """Qualify the real diffusion denoiser then route every sampling call."""
        torch = self.torch
        pipeline = self.pipeline
        source_model = pipeline.model
        qualification_device = torch.device("cuda") if self.cpu_offload_enabled else pipeline.device
        if self.cpu_offload_enabled:
            # The pipeline is deliberately loaded on CPU to avoid overlapping
            # conditioner/DIT/VAE residency. Move only the conditioner for the
            # real probe input; running this 1022px image encoder on CPU took
            # several minutes and does not measure the AMD inference path.
            pipeline.conditioner.to(qualification_device)
        try:
            image = torch.zeros((1, 3, 1022, 1022), dtype=pipeline.dtype, device=qualification_device)
            with torch.inference_mode():
                cond = pipeline.encode_cond(
                    image=image,
                    additional_cond_inputs={},
                    do_classifier_free_guidance=False,
                    dual_guidance=False,
                )
        finally:
            if self.cpu_offload_enabled:
                pipeline.conditioner.to("cpu")
        latents = torch.zeros((1, *pipeline.vae.latent_shape), dtype=pipeline.dtype, device=qualification_device)
        timestep = torch.zeros((1,), dtype=pipeline.dtype, device=qualification_device)
        guidance = torch.full((1,), 5.0, dtype=pipeline.dtype, device=qualification_device)
        reuse_migraphx_rejection = self.cpu_offload_enabled
        _, report = self.runtime.run_region(
            "generate-geometry",
            "hunyuan-miniturbo-denoiser",
            pipeline.model,
            [latents, timestep, cond],
            kwargs={"guidance": guidance},
            prefer_migraphx=not reuse_migraphx_rejection,
            allow_cpu_fallback=False,
            adapter_revision=f"git:{HUNYUAN_SOURCE_REVISION}",
            model_identity=self.model_id,
            weights_identity=f"{HUNYUAN_WEIGHTS_REPO}@{HUNYUAN_WEIGHTS_REVISION}",
            input_artifact_identity=None,
            benchmark_repetitions=3,
            atol=0.005,
            rtol=0.005,
            min_speedup=1.0,
        )
        selected = self.runtime.execution_callable("hunyuan-miniturbo-denoiser", pipeline.model, report)

        class RoutedDenoiser(torch.nn.Module):
            def __init__(self, selected_callable: Any, model_capabilities: Any) -> None:
                super().__init__()
                # Accelerate traverses registered children when installing
                # model-level CPU-offload hooks. Preserve the real source
                # module even when execution routes through its selected
                # runtime callable.
                self.source_model = model_capabilities
                object.__setattr__(self, "_selected_callable", selected_callable)
                _preserve_pipeline_model_controls(model_capabilities, self)

            def forward(self, *args: Any, **kwargs: Any):
                return self._selected_callable(*args, **kwargs)

        pipeline.model = RoutedDenoiser(selected, source_model).to(device=pipeline.device, dtype=pipeline.dtype)
        self.module_report = {
            "module": report.module,
            "backend": report.backend,
            "compile_outcome": report.compile_outcome,
            "fallback_reason": report.fallback_reason,
            "warm_latency_ms": report.latency_ms,
            "baseline_latency_ms": report.baseline_latency_ms,
            "candidate_latency_ms": report.candidate_latency_ms,
            "compile_latency_ms": report.compile_latency_ms,
            "peak_vram_bytes": report.peak_vram_bytes,
            "runtime_versions": report.runtime_versions,
            "routing_basis": (
                "explicit PyTorch ROCm route; same source/weights/device had a slower MIGraphX result in the immediately preceding bounded chair run"
                if reuse_migraphx_rejection else "current MIGraphX qualification"
            ),
            "reused_migraphx_rejection": (
                {
                    "evidence": "api/runtime/adapters/geometry/evidence/hunyuan-offload-chair-run1/runtime-report.json",
                    "compile_outcome": "rejected_slow",
                    "migraphx_latency_ms": 692.8205073333326,
                    "pytorch_rocm_latency_ms": 418.0370923331793,
                    "min_speedup": 1.0,
                    "model_identity": self.model_id,
                    "weights_identity": f"{HUNYUAN_WEIGHTS_REPO}@{HUNYUAN_WEIGHTS_REVISION}",
                    "device": "AMD Radeon RX 7900 GRE / gfx1100",
                }
                if reuse_migraphx_rejection else None
            ),
        }

    def generate(self, source_image: Path, staged_glb: Path, *, seed: int | None, detail_level: str) -> GeometryGenerationResult:
        from PIL import Image
        torch = self.torch
        profiles = {
            "draft": {"num_inference_steps": 20, "octree_resolution": 256, "num_chunks": 8000},
            "balanced": {"num_inference_steps": 50, "octree_resolution": 384, "num_chunks": 8000},
            "high-detail": {"num_inference_steps": 50, "octree_resolution": 512, "num_chunks": 8000},
        }
        if detail_level not in profiles:
            raise ValueError("detail_level must be draft, balanced, or high-detail")
        with Image.open(source_image) as opened:
            image = opened.convert("RGBA" if "A" in opened.getbands() else "RGB")
            generator = torch.Generator(device="cuda")
            if seed is None:
                generator.seed()
                actual_seed = int(generator.initial_seed())
            else:
                generator.manual_seed(seed)
                actual_seed = seed
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            self._write_probe_state(
                phase="generation_started",
                generation_started_epoch=time.time(),
                seed=actual_seed,
                model_load_ms=self.model_load_ms,
                adapter_setup_ms=self.adapter_setup_ms,
                cpu_offload_enabled=self.cpu_offload_enabled,
            )
            with torch.inference_mode():
                mesh = self.pipeline(
                    image=image,
                    generator=generator,
                    output_type="trimesh",
                    mc_algo="mc",
                    enable_pbar=False,
                    **profiles[detail_level],
                )[0]
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            peak_vram_bytes = int(torch.cuda.max_memory_allocated())
        if mesh is None or len(getattr(mesh, "faces", ())) == 0 or len(getattr(mesh, "vertices", ())) < 3:
            raise RuntimeError("Hunyuan did not produce a non-empty triangle mesh")
        staged_glb.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(staged_glb, file_type="glb")
        self._write_probe_state(
            phase="generation_complete",
            seed=actual_seed,
            elapsed_ms=elapsed_ms,
            peak_vram_bytes=peak_vram_bytes,
            vertex_count=len(mesh.vertices),
            triangle_count=len(mesh.faces),
        )
        module_backend = str(self.module_report["backend"])
        runtime_versions = self.module_report["runtime_versions"]
        runtime_name = "PyTorch ROCm" if module_backend == "pytorch_rocm" else "Torch-MIGraphX"
        adapter_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        return GeometryGenerationResult(
            model_id=self.model_id,
            weights_id=f"{HUNYUAN_WEIGHTS_REPO}@{HUNYUAN_WEIGHTS_REVISION}:dit+vae",
            weights_digest=self.weights_digest,
            upstream_repository="https://github.com/Tencent-Hunyuan/Hunyuan3D-2",
            upstream_revision=HUNYUAN_SOURCE_REVISION,
            adapter_id="modly.geometry.hunyuan-mini-turbo",
            adapter_revision=f"sha256:{adapter_digest}",
            backend=module_backend,
            runtime=runtime_name,
            device="AMD Radeon RX 7900 GRE / gfx1100",
            elapsed_ms=elapsed_ms,
            peak_vram_bytes=peak_vram_bytes,
            seed=actual_seed,
            parameters={
                **profiles[detail_level],
                "guidance_scale": 5.0,
                "box_v": 1.01,
                "mc_level": 0.0,
                "mc_algorithm": "scikit-image Lewiner marching cubes on CPU",
                "model_load_ms": self.model_load_ms,
                "neural_module_runtime": self.module_report,
                "pinned_source_archive_sha256": f"sha256:{SOURCE_ARCHIVE_SHA256}",
                "pinned_weight_files": {"dit": f"sha256:{DIT_SHA256}", "vae": f"sha256:{VAE_SHA256}"},
                "detail_level": detail_level,
                "model_cpu_offload": self.cpu_offload_enabled,
                "gpu_budget": self.gpu_budget.as_dict(),
                "seed": actual_seed,
                "unseen_surfaces": "model-inferred, not observed",
            },
        )


def load_reference_adapter():
    """Create the explicitly selected pinned candidate for a Modly extension call."""
    candidate = os.environ.get("MODLY_GEOMETRY_CANDIDATE", "hunyuan-mini-turbo")
    if candidate == "hunyuan-mini-turbo":
        return HunyuanMiniTurboAdapter()
    if candidate == "triposr":
        from runtime.adapters.geometry.triposr_runner import load_reference_adapter as load_triposr
        return load_triposr()
    raise RuntimeError(f"Unknown reference geometry candidate: {candidate}")
