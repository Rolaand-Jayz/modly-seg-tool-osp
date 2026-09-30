"""Pinned P3-SAM AMD adapter boundary.

Model imports happen only after an explicit process invocation. The adapter
requires a project-managed ROCm environment and pre-provisioned, hash-checked
weights; it never downloads code or model files during workflow execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import asdict
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from .regions import CandidateMask, PartSegmentationError


P3SAM_SOURCE_REVISION = "e96be065375438962375b55326416291342958a7"
P3SAM_WEIGHT_SHA256 = "eb76550cfbe06f154c6e9b17167ccfc28222bb4a216ec7b12ac2bf7d762de38c"
P3SAM_WEIGHT_BYTES = 450_968_044
SONATA_WEIGHT_SHA256 = "c5ced5acdae30d1c469713398073a866e25e6e414e23feed5dc025373657ac50"
SONATA_WEIGHT_BYTES = 434_008_287
WEIGHT_MANIFEST_SHA256 = "17f4c933a5e227d4ccdf6ded8726f067684590a86440fe9c45b8c2a2aa35f784"
SONATA_CONFIG_OVERRIDE_SHA256 = "fa2531849a40ae75c776de36619ed2ceb84fcbe837765e31f14305d6ac5b1b32"
SONATA_UPSTREAM_CONFIG_SHA256 = "91c32517a9d5a3c26355ad86aeb19fa179a5acc706ed0b6e9dde39710c7323a5"


@dataclass(frozen=True)
class P3SAMResult:
    masks: tuple[CandidateMask, ...]
    backend: str
    device: str
    runtime_version: str
    latency_ms: float
    peak_vram_allocated_bytes: int
    peak_vram_reserved_bytes: int
    warm_latency_ms: float | None
    warm_masks_match: bool | None
    model_id: str
    weight_identity: str
    runtime_reports: tuple[dict[str, Any], ...]
    compatibility_shims: dict[str, Any]
    sonata_config_override_sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_mask_digest(masks: tuple[CandidateMask, ...]) -> str:
    """Hash unordered part masks by exact canonical face membership."""
    canonical = sorted(tuple(sorted(mask.face_ids)) for mask in masks)
    encoded = json.dumps(canonical, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_clean_revision(root: Path, expected: str, label: str) -> None:
    try:
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise PartSegmentationError("P3SAM_SOURCE_UNVERIFIABLE", f"pinned {label} source revision could not be verified") from exc
    if revision != expected or dirty:
        raise PartSegmentationError("P3SAM_SOURCE_REVISION_MISMATCH", f"{label} source must be clean at pinned revision {expected}")


def verify_installation() -> tuple[Path, Path, Path]:
    """Resolve only explicitly provisioned project adapter files."""
    p3_root_value = os.environ.get("MODLY_P3SAM_SOURCE")
    p3_weight_value = os.environ.get("MODLY_P3SAM_WEIGHTS")
    sonata_weight_value = os.environ.get("MODLY_SONATA_WEIGHTS")
    values = (p3_root_value, p3_weight_value, sonata_weight_value)
    if not all(values):
        raise PartSegmentationError(
            "P3SAM_ADAPTER_NOT_READY",
            "P3-SAM is not provisioned. Install the pinned adapter in Modly's managed AMD environment and set its source and weight paths.",
        )
    p3_root, p3_weights, sonata_weights = (Path(value).expanduser().resolve() for value in values if value)
    if not p3_root.is_dir() or not p3_weights.is_file() or not sonata_weights.is_file():
        raise PartSegmentationError("P3SAM_ADAPTER_NOT_READY", "configured pinned P3-SAM source or checkpoint path is missing")
    _require_clean_revision(p3_root, P3SAM_SOURCE_REVISION, "P3-SAM")
    for path, byte_count, expected_hash, label in (
        (p3_weights, P3SAM_WEIGHT_BYTES, P3SAM_WEIGHT_SHA256, "P3-SAM weights"),
        (sonata_weights, SONATA_WEIGHT_BYTES, SONATA_WEIGHT_SHA256, "Sonata weights"),
    ):
        if path.stat().st_size != byte_count or sha256_file(path) != expected_hash:
            raise PartSegmentationError("P3SAM_WEIGHT_IDENTITY_MISMATCH", f"{label} do not match the pinned byte size and SHA-256")
    return p3_root, p3_weights, sonata_weights


def _assert_amd_rocm(torch: Any) -> str:
    if not torch.cuda.is_available() or not getattr(torch.version, "hip", None):
        raise PartSegmentationError("AMD_ROCM_UNAVAILABLE", "P3-SAM requires the project-managed PyTorch ROCm runtime and an available AMD GPU")
    device = torch.cuda.get_device_name(0)
    if "AMD" not in device.upper() or "7900 GRE" not in device.upper():
        raise PartSegmentationError("UNSUPPORTED_GPU", f"P3-SAM target probe requires RX 7900 GRE; detected {device}")
    if getattr(torch.version, "cuda", None):
        raise PartSegmentationError("CUDA_RUNTIME_FORBIDDEN", "P3-SAM adapter environment must use ROCm PyTorch without a CUDA runtime")
    return device


def sonata_config_override(source_root: Path) -> tuple[dict[str, Any], str]:
    """Verify and load the adapter-local attention override for pinned Sonata."""
    override_path = Path(__file__).with_name("SONATA_CONFIG_OVERRIDE.json")
    if sha256_file(override_path) != SONATA_CONFIG_OVERRIDE_SHA256:
        raise PartSegmentationError("SONATA_CONFIG_OVERRIDE_IDENTITY_MISMATCH", "adapter Sonata override file does not match its pinned SHA-256")
    upstream_config = source_root / "XPart" / "partgen" / "config" / "sonata.json"
    if not upstream_config.is_file() or sha256_file(upstream_config) != SONATA_UPSTREAM_CONFIG_SHA256:
        raise PartSegmentationError("SONATA_UPSTREAM_CONFIG_IDENTITY_MISMATCH", "pinned Sonata source config does not match the audited byte identity")
    document = json.loads(override_path.read_text(encoding="utf-8"))
    if document.get("upstream", {}).get("source_revision") != P3SAM_SOURCE_REVISION:
        raise PartSegmentationError("SONATA_CONFIG_REVISION_MISMATCH", "adapter override targets a different Sonata source revision")
    if document.get("upstream", {}).get("config_sha256") != SONATA_UPSTREAM_CONFIG_SHA256:
        raise PartSegmentationError("SONATA_UPSTREAM_CONFIG_IDENTITY_MISMATCH", "adapter override targets a different upstream Sonata config")
    overrides = document.get("overrides")
    if overrides != {"enable_flash": False}:
        raise PartSegmentationError("SONATA_CONFIG_OVERRIDE_UNSUPPORTED", "only the reviewed pure-PyTorch attention override is supported")
    return overrides, SONATA_CONFIG_OVERRIDE_SHA256


def install_p3sam_compatibility_shims() -> dict[str, Any]:
    """Install the exact adapter-local namespaces needed by pinned Sonata."""
    from .sonata_spconv_compat import install_spconv_compat
    from .sonata_scatter_compat import install_torch_scatter_compat
    from .sonata_dict_compat import install_sonata_dict_compat
    from .sonata_timm_compat import install_sonata_timm_compat

    spconv_module = install_spconv_compat()
    scatter_module = install_torch_scatter_compat()
    dict_module = install_sonata_dict_compat()
    timm_module, timm_layers = install_sonata_timm_compat()
    return {
        "spconv": {
            "provider": "modly.adapter.sonata_spconv_compat",
            "module_file": str(getattr(spconv_module, "__file__", __file__)),
            "apis": ["SparseConvTensor", "SparseConvTensor.replace_feature", "SubMConv3d", "modules.is_spconv_module"],
            "unsupported": ["non-SubM sparse convolution", "stride other than 1", "dilation other than 1", "groups other than 1", "even kernels", "indice caching", "training/backward qualification"],
        },
        "torch_scatter": {
            "provider": "modly.adapter.sonata_scatter_compat",
            "module_file": str(getattr(scatter_module, "__file__", __file__)),
            "apis": ["segment_csr"],
            "unsupported": ["other torch_scatter operators", "nonzero reduction dimensions", "empty CSR segments", "native CUDA/C++ ABI compatibility"],
        },
        "sonata_dict": {
            "provider": "modly.adapter.sonata_dict_compat",
            "import_namespace": "addict.Dict",
            "module_file": str(getattr(dict_module, "__file__", __file__)),
            "apis": ["mapping construction", "attribute/item read and write", "attribute deletion", "recursive mappings in mappings/lists/tuples"],
            "unsupported": ["unobserved upstream addict APIs", "full upstream package compatibility", "attribute access for mapping-method name collisions"],
        },
        "timm_drop_path": {
            "provider": "modly.adapter.sonata_timm_compat",
            "import_namespace": "timm.layers.DropPath",
            "module_file": str(getattr(timm_layers, "__file__", __file__)),
            "parent_module_file": str(getattr(timm_module, "__file__", __file__)),
            "apis": ["DropPath(drop_prob=0.0, scale_by_keep=True)", "forward(Tensor)", "extra_repr"],
            "unsupported": ["all other timm APIs", "full upstream timm package compatibility", "batched logical-sample grouping for flattened Sonata point features"],
            "train_behavior": "per-leading-row Bernoulli mask; scale kept rows by inverse keep probability; eval and zero drop probability are identity",
        },
    }


def _import_p3sam_modules(source_root: Path, module_importer=None, source_overlay_installer=None, auto_mask_overlay_installer=None) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    """Install adapter shims before importing upstream Sonata/P3-SAM modules."""
    import_module = module_importer or importlib.import_module
    sys.path.insert(0, str(source_root / "XPart" / "partgen"))
    sys.path.insert(0, str(source_root / "P3-SAM"))
    sys.path.insert(0, str(source_root / "P3-SAM" / "demo"))
    torch = import_module("torch")
    shims = install_p3sam_compatibility_shims()
    from .sonata_transform_overlay import install_sonata_transform_overlay

    install_overlay = source_overlay_installer or install_sonata_transform_overlay
    shims["sonata_transform"] = install_overlay(source_root)
    sonata = import_module("models.sonata")
    from .p3sam_auto_mask_overlay import install_p3sam_auto_mask_overlay

    install_auto_mask_overlay = auto_mask_overlay_installer or install_p3sam_auto_mask_overlay
    shims["p3sam_auto_mask"] = install_auto_mask_overlay(source_root)
    auto_mask = import_module("auto_mask")
    return torch, sonata, auto_mask.AutoMask, auto_mask.set_seed, shims


def run_p3sam(
    mesh: Any,
    *,
    point_num: int = 10_000,
    prompt_num: int = 32,
    prompt_batch_size: int = 4,
    seed: int = 42,
    workspace_dir: Path | None = None,
    input_artifact_identity: str | None = None,
    measure_warm_pass: bool = False,
    prefer_migraphx: bool = True,
) -> P3SAMResult:
    """Infer face masks with fixed model/weights and a bounded first-pass preset."""
    if type(point_num) is not int or point_num < 1 or type(prompt_num) is not int or prompt_num < 1:
        raise PartSegmentationError("INVALID_SEGMENTATION_PARAMETERS", "point_num and prompt_num must be positive integers")
    if type(prompt_batch_size) is not int or prompt_batch_size < 1:
        raise PartSegmentationError("INVALID_SEGMENTATION_PARAMETERS", "prompt_batch_size must be a positive integer")
    if type(measure_warm_pass) is not bool:
        raise PartSegmentationError("INVALID_SEGMENTATION_PARAMETERS", "measure_warm_pass must be a boolean")
    if type(prefer_migraphx) is not bool:
        raise PartSegmentationError("INVALID_SEGMENTATION_PARAMETERS", "prefer_migraphx must be a boolean")
    p3_root, p3_weights, sonata_weights = verify_installation()
    try:
        torch, p3_sonata, AutoMask, set_seed, compatibility_shims = _import_p3sam_modules(p3_root)
        custom_config, config_identity = sonata_config_override(p3_root)
    except PartSegmentationError:
        raise
    except Exception as exc:
        raise PartSegmentationError("P3SAM_IMPORT_FAILED", f"pinned P3-SAM/Sonata AMD dependencies could not be imported: {type(exc).__name__}: {exc}") from exc
    device = _assert_amd_rocm(torch)
    from services.amd_runtime import AMDInferenceRuntime

    log_path = (
        workspace_dir / "StructuredAssets" / "Runtime" / "p3sam-amd-runtime.jsonl"
        if workspace_dir is not None
        else Path(os.environ.get("MODLY_AMD_RUNTIME_LOG", "p3sam-amd-runtime.jsonl"))
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    runtime = AMDInferenceRuntime(detail_log=log_path)
    runtime_reports: list[dict[str, Any]] = []
    peak_allocated = 0
    peak_reserved = 0

    # Upstream currently hard-codes `/root/sonata` as a download directory.
    # Replace that lookup only within this process with the hash-verified local
    # checkpoint so a workflow cannot write to a user's or system cache.
    original_sonata_load = p3_sonata.load

    def pinned_sonata_load(*_args: Any, **_kwargs: Any) -> Any:
        return original_sonata_load(str(sonata_weights), custom_config=custom_config)

    p3_sonata.load = pinned_sonata_load
    try:
        model = AutoMask(
            ckpt_path=str(p3_weights),
            point_num=point_num,
            prompt_num=prompt_num,
            threshold=0.95,
            post_process=True,
        )
    except Exception as exc:
        raise PartSegmentationError("P3SAM_MODEL_LOAD_FAILED", f"pinned P3-SAM model could not load on AMD ROCm: {type(exc).__name__}: {exc}") from exc
    finally:
        p3_sonata.load = original_sonata_load

    try:
        # Include resident weights and any transient load-time peak. Region
        # benchmarking resets allocator counters later, so record the loader
        # peak before the first dense module is qualified.
        peak_allocated = int(torch.cuda.max_memory_allocated())
        peak_reserved = int(torch.cuda.max_memory_reserved())

        class QualifiedRegion(torch.nn.Module):
            def __init__(self, module_name: str, module: Any) -> None:
                super().__init__()
                self.module_name = module_name
                self.module = module
                self.selected: Any = None

            def forward(self, *args: Any, **kwargs: Any) -> Any:
                nonlocal peak_allocated, peak_reserved
                if self.selected is not None:
                    return self.selected(*args, **kwargs)
                output, report = runtime.run_region(
                    "reference-part-segmentation",
                    self.module_name,
                    self.module,
                    args,
                    kwargs=kwargs,
                    prefer_migraphx=prefer_migraphx,
                    adapter_revision="builtin:1.0.0",
                    model_identity=f"tencent/Hunyuan3D-Part:P3-SAM@git:{P3SAM_SOURCE_REVISION}",
                    weights_identity=(
                        f"manifest:sha256:{WEIGHT_MANIFEST_SHA256};"
                        f"p3sam:sha256:{P3SAM_WEIGHT_SHA256};sonata:sha256:{SONATA_WEIGHT_SHA256}"
                    ),
                    input_artifact_identity=input_artifact_identity,
                    benchmark_repetitions=3,
                )
                self.selected = runtime.execution_callable(self.module_name, self.module, report)
                runtime_reports.append(asdict(report))
                peak_allocated = max(peak_allocated, int(torch.cuda.max_memory_allocated()))
                peak_reserved = max(peak_reserved, int(torch.cuda.max_memory_reserved()))
                return output

        for region_name in (
            "mlp",
            "seg_mlp_1", "seg_mlp_2", "seg_mlp_3",
            "seg_s2_mlp_g", "seg_s2_mlp_1", "seg_s2_mlp_2", "seg_s2_mlp_3",
            "iou_mlp", "iou_mlp_out",
        ):
            setattr(model.model, region_name, QualifiedRegion(f"p3sam.{region_name}", getattr(model.model, region_name)))
        def infer_pass() -> tuple[tuple[CandidateMask, ...], float]:
            set_seed(seed)
            torch.cuda.synchronize()
            started = time.perf_counter()
            _aabbs, raw_face_ids, result_mesh = model.predict_aabb(
                mesh,
                point_num=point_num,
                prompt_num=prompt_num,
                threshold=0.95,
                post_process=True,
                show_info=False,
                save_mid_res=False,
                clean_mesh_flag=False,
                seed=seed,
                is_parallel=False,
                prompt_bs=prompt_batch_size,
            )
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            labels = raw_face_ids.tolist() if hasattr(raw_face_ids, "tolist") else list(raw_face_ids)
            if (len(labels) != len(mesh.faces) or len(result_mesh.faces) != len(mesh.faces)
                    or not (result_mesh.faces == mesh.faces).all()):
                raise PartSegmentationError(
                    "P3SAM_TOPOLOGY_CHANGED",
                    "P3-SAM changed or reordered input faces; mappings cannot be transferred to the canonical topology",
                )
            if any(type(label) is not int for label in labels):
                raise PartSegmentationError("P3SAM_INVALID_FACE_LABEL", "P3-SAM returned non-integer face part labels")
            unique_labels = sorted(set(labels))
            if any(label < 0 for label in unique_labels):
                raise PartSegmentationError("P3SAM_UNASSIGNED_FACES", "P3-SAM left faces without a part assignment under the required complete-partition policy")
            masks = tuple(
                CandidateMask(tuple(index for index, value in enumerate(labels) if value == label))
                for label in unique_labels
            )
            return masks, elapsed_ms

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        masks, latency_ms = infer_pass()
        warm_latency_ms: float | None = None
        warm_masks_match: bool | None = None
        if measure_warm_pass:
            warm_masks, warm_latency_ms = infer_pass()
            warm_masks_match = canonical_mask_digest(masks) == canonical_mask_digest(warm_masks)
            masks = warm_masks
        if not runtime_reports:
            raise PartSegmentationError(
                "P3SAM_RUNTIME_REGION_UNQUALIFIED",
                "P3-SAM completed without qualifying a dense PyTorch inference region through the AMD Runtime",
            )
        selected_backends = sorted({str(report["backend"]) for report in runtime_reports})
        backend = selected_backends[0] if len(selected_backends) == 1 else "+".join(selected_backends)
        runtime_description = json.dumps(
            runtime_reports[0]["runtime_versions"] if runtime_reports else {
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "rocm_hip": torch.version.hip,
            },
            sort_keys=True,
        )
        return P3SAMResult(
            masks=masks,
            backend=backend,
            device=device,
            runtime_version=runtime_description,
            latency_ms=latency_ms,
            peak_vram_allocated_bytes=max(peak_allocated, int(torch.cuda.max_memory_allocated())),
            peak_vram_reserved_bytes=max(peak_reserved, int(torch.cuda.max_memory_reserved())),
            warm_latency_ms=warm_latency_ms,
            warm_masks_match=warm_masks_match,
            model_id="tencent/Hunyuan3D-Part:P3-SAM",
            weight_identity=(
                f"manifest:sha256:{WEIGHT_MANIFEST_SHA256};"
                f"p3sam:sha256:{P3SAM_WEIGHT_SHA256};sonata:sha256:{SONATA_WEIGHT_SHA256}"
            ),
            runtime_reports=tuple(runtime_reports),
            compatibility_shims=compatibility_shims,
            sonata_config_override_sha256=config_identity,
        )
    except PartSegmentationError:
        raise
    except Exception as exc:
        raise PartSegmentationError("P3SAM_INFERENCE_FAILED", f"P3-SAM failed during native 3D segmentation: {type(exc).__name__}: {exc}") from exc
    finally:
        del model
        runtime.close()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
