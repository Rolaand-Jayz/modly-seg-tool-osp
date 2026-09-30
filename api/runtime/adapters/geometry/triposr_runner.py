"""Pinned TripoSR adapter with AMD routing and adapter-local CPU surface extraction."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import time
from typing import Any

from runtime.adapters.geometry.stage import GeometryGenerationResult


TRIPOSR_SOURCE_REVISION = "107cefdc244c39106fa830359024f6a2f1c78871"
TRIPOSR_WEIGHTS_REVISION = "9700b06c1641864ecbbe5eb0d89b967f3045cd5e"
TRIPOSR_SOURCE_ARCHIVE_SHA256 = "bcb414550dcfcb9f5ea6a7b9c12f2bbff889f5b4a564a493178393360d9034ec"
TRIPOSR_CHECKPOINT_SHA256 = "429e2c6b22a0923967459de24d67f05962b235f79cde6b032aa7ed2ffcd970ee"
TRIPOSR_CONFIG_SHA256 = "74ca708ce086bf68e97709ea6b3d91f14717921c04691e84043f0eb8fcc68e62"
TRIPOSR_UPSTREAM_UTILS_SHA256 = "d9d1a645fef7ea489615bff3471fd01a4941a2c0e138c5dff5daae619563279b"
DINO_CONFIG_SHA256 = "b87c0270b97db085fd82cf114a761fd0f62ae7914fbd407c752a2260646b689c"
DINO_CONFIG_REVISION = "f567a076ac86157a72e04dba41b4d70a82c24d00"
TRIPOSR_REPOSITORY = "VAST-AI-Research/TripoSR"
TRIPOSR_MODEL_REPOSITORY = "stabilityai/TripoSR"
DINO_REPOSITORY = "facebook/dino-vitb16"


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
    container_root = Path("/models/triposr")
    if container_root.is_dir():
        return container_root
    project_root = Path(__file__).resolve().parents[4]
    return project_root / ".modly-amd-runtime" / "models" / "triposr"


def _ensure_project_migraphx_python_path() -> None:
    """Expose the image-installed MIGraphX Python package inside nested workers."""
    rocm_python_path = Path("/opt/rocm/lib")
    if rocm_python_path.is_dir() and str(rocm_python_path) not in sys.path:
        sys.path.append(str(rocm_python_path))


def _safe_extract_pinned_source(archive: Path, destination: Path) -> Path:
    if not archive.is_file() or _sha256(archive) != TRIPOSR_SOURCE_ARCHIVE_SHA256:
        raise RuntimeError("Pinned TripoSR source archive is missing or does not match its SHA-256 identity")
    source_root = destination / "adapter-source"
    # Recreate only this generated adapter overlay from the archive so provenance
    # always hashes the untouched upstream file before the reviewed local patch.
    shutil.rmtree(source_root, ignore_errors=True)
    temporary = destination / ".adapter-source.partial"
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as package:
        members = package.getmembers()
        for member in members:
            resolved = (temporary / member.name).resolve()
            if resolved != temporary.resolve() and temporary.resolve() not in resolved.parents:
                raise RuntimeError("Pinned TripoSR source archive contains an unsafe path")
            if member.issym() or member.islnk() or member.isdev():
                raise RuntimeError("Pinned TripoSR source archive contains an unsupported link or device entry")
        package.extractall(temporary, members=members, filter="data")
    candidates = [path for path in temporary.iterdir() if path.is_dir()]
    if len(candidates) != 1 or not (candidates[0] / "tsr" / "system.py").is_file():
        raise RuntimeError("Pinned TripoSR archive does not have the expected source layout")
    candidates[0].rename(source_root)
    shutil.rmtree(temporary, ignore_errors=True)
    return source_root


def _prepare_source(root: Path) -> tuple[Path, str, str]:
    source = _safe_extract_pinned_source(root / "source.tar.gz", root)
    utils = source / "tsr" / "utils.py"
    original = utils.read_text(encoding="utf-8")
    upstream_utils_digest = hashlib.sha256(original.encode("utf-8")).hexdigest()
    if upstream_utils_digest != TRIPOSR_UPSTREAM_UTILS_SHA256:
        raise RuntimeError("Pinned TripoSR upstream utility source has a SHA-256 mismatch")
    patched_import = "try:\n    import rembg\nexcept ImportError:\n    rembg = None\n"
    if patched_import not in original:
        old = "import rembg\n"
        if old not in original:
            raise RuntimeError("TripoSR rembg source import differs from the reviewed adapter patch")
        original = original.replace(old, patched_import, 1)
    guard = "    if rembg is None:\n        raise RuntimeError(\"rembg is required for this background-removal path; geometry fixtures must provide alpha or a neutral-gray background\")\n"
    function_header = "def remove_background(\n"
    if guard not in original:
        start = original.find(function_header)
        if start < 0:
            raise RuntimeError("TripoSR background-removal function differs from the reviewed adapter patch")
        header_end = original.find("\n) -> PIL.Image.Image:\n", start)
        if header_end < 0:
            raise RuntimeError("TripoSR background-removal signature differs from the reviewed adapter patch")
        insert_at = header_end + len("\n) -> PIL.Image.Image:\n")
        original = original[:insert_at] + guard + original[insert_at:]
    utils.write_text(original, encoding="utf-8")
    patch_digest = _sha256(utils)
    return source, upstream_utils_digest, patch_digest


def _add_dependencies(root: Path) -> tuple[Path, str, str]:
    candidate_packages = root / "python-packages"
    shared_packages = root.parent / "hunyuan-mini-turbo" / "python-packages"
    if not candidate_packages.is_dir():
        raise RuntimeError("TripoSR candidate-local dependencies have not been installed")
    if not shared_packages.is_dir():
        raise RuntimeError("Pinned project-local shared vision dependencies are missing")
    source_root, upstream_utils_digest, source_patch_digest = _prepare_source(root)
    adapter_helpers = Path(__file__).resolve().parent / "triposr_cpu_ops"
    for package in (str(candidate_packages), str(shared_packages), str(adapter_helpers), str(source_root)):
        if package in sys.path:
            sys.path.remove(package)
        sys.path.insert(0, package)
    os.environ["HF_HUB_OFFLINE"] = "1"
    return source_root, upstream_utils_digest, source_patch_digest


def _verify_candidate(root: Path) -> tuple[Path, str, str, str]:
    source_root, upstream_utils_digest, source_patch_digest = _add_dependencies(root)
    checkpoint = root / "weights" / "model.ckpt"
    config = root / "weights" / "config.yaml"
    dino_config = root / "weights" / f"facebook-dino-vitb16-{DINO_CONFIG_REVISION}" / "config.json"
    for path, expected, label in (
        (checkpoint, TRIPOSR_CHECKPOINT_SHA256, "checkpoint"),
        (config, TRIPOSR_CONFIG_SHA256, "config"),
        (dino_config, DINO_CONFIG_SHA256, "DINO config"),
    ):
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"Pinned TripoSR {label} is missing or has a SHA-256 mismatch")
    combined = hashlib.sha256(
        f"checkpoint:{TRIPOSR_CHECKPOINT_SHA256}\nconfig:{TRIPOSR_CONFIG_SHA256}\ndino:{DINO_CONFIG_SHA256}\n".encode("ascii")
    ).hexdigest()
    return source_root, f"sha256:{combined}", upstream_utils_digest, source_patch_digest


class TripoSRAdapter:
    """Single-image reconstruction adapter using pinned TripoSR weights."""

    model_id = "stabilityai.TripoSR"

    def _write_probe_state(self, **updates: object) -> None:
        state_path = os.environ.get("MODLY_GEOMETRY_PROBE_STATE_FILE")
        if not state_path:
            return
        torch = getattr(self, "torch", None)
        state: dict[str, object] = {
            "monotonic": time.monotonic(),
            "model_loaded": bool(getattr(self, "model_load_ms", None)),
            "torch_allocated_bytes": int(torch.cuda.memory_allocated()) if torch is not None else None,
            "torch_reserved_bytes": int(torch.cuda.memory_reserved()) if torch is not None else None,
        }
        state.update(updates)
        Path(state_path).write_text(json.dumps(state, sort_keys=True) + "\n", encoding="utf-8")

    def __init__(self) -> None:
        import torch

        if not getattr(torch.version, "hip", None) or not torch.cuda.is_available():
            raise RuntimeError("TripoSR generation requires the project ROCm container and an available AMD HIP device")
        self.torch = torch
        self.root = _candidate_root()
        (self.source_root, self.weights_digest, self.upstream_utils_digest,
         self.source_patch_digest) = _verify_candidate(self.root)
        from services.amd_runtime import AMDInferenceRuntime
        from tsr.system import TSR
        import tsr.models.tokenizers.image as image_tokenizer

        dino_config = self.root / "weights" / f"facebook-dino-vitb16-{DINO_CONFIG_REVISION}" / "config.json"
        original_download = image_tokenizer.hf_hub_download

        def pinned_dino_config(*, repo_id: str, filename: str, **kwargs: Any) -> str:
            if repo_id == DINO_REPOSITORY and filename == "config.json":
                return str(dino_config)
            raise RuntimeError(f"Unpinned Hugging Face file request rejected: {repo_id}/{filename}")

        image_tokenizer.hf_hub_download = pinned_dino_config
        try:
            self.load_started = time.perf_counter()
            self.model = TSR.from_pretrained(
                str(self.root / "weights"), config_name="config.yaml", weight_name="model.ckpt"
            )
        finally:
            image_tokenizer.hf_hub_download = original_download
        self.model_load_ms = (time.perf_counter() - self.load_started) * 1000.0
        self.model.eval().to(device="cuda")
        self.model.renderer.set_chunk_size(8192)
        _ensure_project_migraphx_python_path()
        self.runtime = AMDInferenceRuntime(detail_log=Path(os.environ.get(
            "MODLY_AMD_RUNTIME_LOG", str(self.root / "runtime" / "amd-runtime.jsonl")
        )))
        self._write_probe_state(model_loaded=True, model_load_ms=self.model_load_ms, phase="backend_qualification")
        self.module_report = self._route_backbone()

    def _route_backbone(self) -> dict[str, object]:
        torch = self.torch
        batch = 1
        tokens = torch.zeros((batch, 1024, 3072), dtype=torch.float32, device="cuda")
        image_tokens = torch.zeros((batch, 1025, 768), dtype=torch.float32, device="cuda")
        _, report = self.runtime.run_region(
            "generate-geometry", "triposr-backbone", self.model.backbone,
            [tokens], kwargs={"encoder_hidden_states": image_tokens},
            prefer_migraphx=True, allow_cpu_fallback=False,
            adapter_revision=f"git:{TRIPOSR_SOURCE_REVISION}", model_identity=self.model_id,
            weights_identity=f"{TRIPOSR_MODEL_REPOSITORY}@{TRIPOSR_WEIGHTS_REVISION}",
            input_artifact_identity=None, benchmark_repetitions=3, atol=0.005, rtol=0.005,
            min_speedup=1.0,
        )
        selected = self.runtime.execution_callable("triposr-backbone", self.model.backbone, report)

        class RoutedBackbone(torch.nn.Module):
            def __init__(self, callable_: Any) -> None:
                super().__init__()
                object.__setattr__(self, "_callable", callable_)

            def forward(self, *args: Any, **kwargs: Any):
                return self._callable(*args, **kwargs)

        self.model.backbone = RoutedBackbone(selected).to(device="cuda")
        return {
            "module": report.module, "backend": report.backend,
            "compile_outcome": report.compile_outcome, "fallback_reason": report.fallback_reason,
            "warm_latency_ms": report.latency_ms, "baseline_latency_ms": report.baseline_latency_ms,
            "candidate_latency_ms": report.candidate_latency_ms, "compile_latency_ms": report.compile_latency_ms,
            "peak_vram_bytes": report.peak_vram_bytes, "runtime_versions": report.runtime_versions,
        }

    def generate(self, source_image: Path, staged_glb: Path, *, seed: int | None, detail_level: str) -> GeometryGenerationResult:
        from PIL import Image
        torch = self.torch
        if detail_level not in {"draft", "balanced", "high-detail"}:
            raise ValueError("detail_level must be draft, balanced, or high-detail")
        resolution = {"draft": 128, "balanced": 256, "high-detail": 384}[detail_level]
        with Image.open(source_image) as opened:
            image = opened.convert("RGBA" if "A" in opened.getbands() else "RGB")
            if "A" in image.getbands():
                from tsr.utils import resize_foreground
                # These fixed alpha fixtures already contain foreground evidence;
                # the rembg inference path is neither needed nor imported.
                image = resize_foreground(image, 0.85)
                rgba = torch.from_numpy(__import__("numpy").asarray(image).copy()).float().div_(255.0)
                rgb = rgba[..., :3] * rgba[..., 3:4] + (1.0 - rgba[..., 3:4]) * 0.5
                image = Image.fromarray((rgb.mul(255).clamp(0, 255).byte().cpu().numpy()))
            generator = torch.Generator(device="cuda")
            if seed is None:
                generator.seed()
                actual_seed = int(generator.initial_seed())
            else:
                generator.manual_seed(seed)
                actual_seed = seed
            torch.manual_seed(actual_seed)
            torch.cuda.manual_seed_all(actual_seed)
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            self._write_probe_state(phase="generation_started", seed=actual_seed, model_load_ms=self.model_load_ms)
            with torch.inference_mode():
                scene_codes = self.model([image], device="cuda")
                self._write_probe_state(phase="scene_codes_ready", seed=actual_seed,
                    torch_peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                    torch_peak_reserved_bytes=int(torch.cuda.max_memory_reserved()))
                mesh = self.model.extract_mesh(scene_codes, has_vertex_color=True, resolution=resolution)[0]
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            peak_vram_bytes = int(torch.cuda.max_memory_allocated())
            peak_reserved_bytes = int(torch.cuda.max_memory_reserved())
            self._write_probe_state(phase="generation_complete", seed=actual_seed, elapsed_ms=elapsed_ms,
                torch_peak_allocated_bytes=peak_vram_bytes, torch_peak_reserved_bytes=peak_reserved_bytes)
        if mesh is None or len(mesh.faces) == 0 or len(mesh.vertices) < 3:
            raise RuntimeError("TripoSR did not produce a non-empty triangle mesh")
        staged_glb.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(staged_glb, file_type="glb")
        backend = str(self.module_report["backend"])
        adapter_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        return GeometryGenerationResult(
            model_id=self.model_id,
            weights_id=f"{TRIPOSR_MODEL_REPOSITORY}@{TRIPOSR_WEIGHTS_REVISION}",
            weights_digest=self.weights_digest,
            upstream_repository="https://github.com/VAST-AI-Research/TripoSR",
            upstream_revision=TRIPOSR_SOURCE_REVISION,
            adapter_id="modly.geometry.triposr",
            adapter_revision=f"sha256:{adapter_digest}",
            backend=backend,
            runtime="PyTorch ROCm" if backend == "pytorch_rocm" else "Torch-MIGraphX",
            device="AMD Radeon RX 7900 GRE / gfx1100",
            elapsed_ms=elapsed_ms,
            peak_vram_bytes=max(peak_vram_bytes, int(self.module_report.get("peak_vram_bytes") or 0)),
            seed=actual_seed,
            parameters={
                "cond_image_size": 512, "renderer_chunk_size": 8192,
                "marching_cubes_resolution": resolution, "marching_cubes": "scikit-image Lewiner CPU",
                "model_load_ms": self.model_load_ms, "neural_module_runtime": self.module_report,
                "torch_peak_reserved_bytes": peak_reserved_bytes,
                "pinned_source_archive_sha256": f"sha256:{TRIPOSR_SOURCE_ARCHIVE_SHA256}",
                "pinned_checkpoint_sha256": f"sha256:{TRIPOSR_CHECKPOINT_SHA256}",
                "pinned_config_sha256": f"sha256:{TRIPOSR_CONFIG_SHA256}",
                "pinned_dino_config_revision": DINO_CONFIG_REVISION,
                "pinned_dino_config_sha256": f"sha256:{DINO_CONFIG_SHA256}",
                "adapter_source_overlay_utils_sha256": f"sha256:{self.source_patch_digest}",
                "upstream_utils_sha256": f"sha256:{self.upstream_utils_digest}",
                "seed": actual_seed, "detail_level": detail_level,
                "background_removal": "not installed; fixtures use alpha compositing or pinned neutral-gray background",
                "unseen_surfaces": "model-inferred, not observed",
            },
        )


def load_reference_adapter() -> TripoSRAdapter:
    return TripoSRAdapter()
