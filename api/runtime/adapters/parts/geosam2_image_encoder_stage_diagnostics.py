"""Payload-free hooks for locating first-use differences inside GeoSAM2 Hiera."""
from __future__ import annotations

from typing import Any

from . import geosam2_lifecycle_diagnostics as lifecycle

SCHEMA = "modly.ticket04.geosam2-image-encoder-stage-diagnostics/1"
MAX_BLOCKS = 32
EXPECTED_BLOCK_COUNT = 24


class ImageEncoderStageCaptureError(RuntimeError):
    """The pinned image encoder does not match the bounded capture contract."""


class ImageEncoderStageCapture:
    """Hash selected Hiera intermediate tensors without retaining their values."""

    def __init__(self, model: Any, torch_module: Any) -> None:
        image_encoder = getattr(model, "image_encoder", None)
        trunk = getattr(image_encoder, "trunk", None)
        patch_embed = getattr(trunk, "patch_embed", None)
        blocks = getattr(trunk, "blocks", None)
        try:
            block_count = len(blocks)
        except (TypeError, AttributeError):
            block_count = 0
        if (trunk is None or not callable(getattr(patch_embed, "register_forward_hook", None))
                or block_count != EXPECTED_BLOCK_COUNT or block_count > MAX_BLOCKS):
            raise ImageEncoderStageCaptureError("pinned image encoder Hiera stages are unavailable")
        self._torch = torch_module
        self._trunk = trunk
        self._roles = lifecycle.CALL_ROLES
        self._role_index = 0
        self._active_role: str | None = None
        self._records: dict[str, list[dict[str, Any]]] = {role: [] for role in self._roles}
        self._handles = []
        self._block_count = block_count
        self._original_pos_embed = getattr(trunk, "_get_pos_embed", None)
        if not callable(self._original_pos_embed):
            raise ImageEncoderStageCaptureError("pinned Hiera positional embedding method is unavailable")
        self._had_instance_pos_embed = "_get_pos_embed" in getattr(trunk, "__dict__", {})
        self._instance_pos_embed = getattr(trunk, "__dict__", {}).get("_get_pos_embed")
        register_pre = getattr(image_encoder, "register_forward_pre_hook", None)
        if not callable(register_pre):
            raise ImageEncoderStageCaptureError("image encoder cannot signal lifecycle roles")
        self._handles.append(register_pre(self._begin_role))
        self._handles.append(patch_embed.register_forward_hook(self._module_output("trunk.patch_embed.output")))
        for index, block in enumerate(blocks):
            register_hook = getattr(block, "register_forward_hook", None)
            if not callable(register_hook):
                self.close()
                raise ImageEncoderStageCaptureError(f"Hiera block {index} cannot register a capture hook")
            self._handles.append(block.register_forward_hook(self._block_hook(index)))

        def observed_pos_embed(*args: Any, **kwargs: Any) -> Any:
            output = self._original_pos_embed(*args, **kwargs)
            self._record("trunk.positional_embedding.output", output)
            return output

        self._observed_pos_embed = observed_pos_embed
        trunk._get_pos_embed = observed_pos_embed

    def _begin_role(self, _module: Any, _inputs: Any) -> None:
        if self._active_role is not None or self._role_index >= len(self._roles):
            raise ImageEncoderStageCaptureError("image encoder call count differs from the locked three-role sequence")
        self._active_role = self._roles[self._role_index]
        self._role_index += 1

    def _record(self, stage: str, value: Any) -> None:
        if self._active_role is None:
            raise ImageEncoderStageCaptureError("Hiera stage ran outside an expected image encoder call")
        summary = lifecycle.digest_tree(value, self._torch)
        self._records[self._active_role].append({"stage": stage, **summary})

    def _module_output(self, stage: str):
        def capture(_module: Any, _inputs: Any, output: Any) -> None:
            self._record(stage, output)
        return capture

    def _block_hook(self, index: int):
        def capture(_module: Any, inputs: Any, output: Any) -> None:
            if index == 0:
                self._record("trunk.blocks.0.input_after_position_add", inputs)
            self._record(f"trunk.blocks.{index}.output", output)
            if index == self._block_count - 1:
                self._active_role = None
        return capture

    def report(self, helper_sha256: str, lock_sha256: str) -> dict[str, Any]:
        expected_block_count = len(getattr(self._trunk, "blocks", ()))
        expected = ["trunk.patch_embed.output", "trunk.positional_embedding.output",
                    "trunk.blocks.0.input_after_position_add"]
        expected.extend(f"trunk.blocks.{index}.output" for index in range(expected_block_count))
        calls = []
        for role in self._roles:
            events = self._records[role]
            calls.append({"role": role,
                          "state": "complete" if [item["stage"] for item in events] == expected else "partial",
                          "events": events,
                          "missing_stages": [stage for stage in expected
                                             if stage not in [item["stage"] for item in events]]})
        complete = self._role_index == len(self._roles) and all(call["state"] == "complete" for call in calls)
        return {"schema": SCHEMA, "state": "complete" if complete else "partial",
                "acceptance_status": "not_assessed",
                "helper_sha256": helper_sha256, "lock_sha256": lock_sha256,
                "roles": calls, "captured_block_count": expected_block_count,
                "caps": {"lifecycle_roles": len(self._roles), "blocks": MAX_BLOCKS,
                         "tensor_byte_limit": lifecycle.MAX_TENSOR_BYTES,
                         "tensor_count_per_stage": lifecycle.MAX_TENSORS_PER_CAPTURE},
                "payloads_persisted": False,
                "interpretation_limit": "Hooks hash transient tensor copies and synchronize GPU work; timing and execution are diagnostic-only.",
                "truth_access": "none; no truth file was mounted"}

    def close(self) -> None:
        self._active_role = None
        current = getattr(self._trunk, "_get_pos_embed", None)
        if current is getattr(self, "_observed_pos_embed", None):
            if self._had_instance_pos_embed:
                self._trunk._get_pos_embed = self._instance_pos_embed
            else:
                delattr(self._trunk, "_get_pos_embed")
        for handle in reversed(self._handles):
            remove = getattr(handle, "remove", None)
            if callable(remove):
                remove()
        self._handles.clear()


def run_with_image_encoder_stages(*, lifecycle_runner: Any, helper_sha256: str,
                                  lock_sha256: str,
                                  on_report: Any | None = None,
                                  **kwargs: Any) -> dict[str, Any]:
    capture = ImageEncoderStageCapture(kwargs["model"], kwargs["torch_module"])
    try:
        result = lifecycle_runner(**kwargs)
        report = capture.report(helper_sha256, lock_sha256)
        result["image_encoder_internal_stages"] = report
        if callable(on_report):
            on_report(report)
        return result
    except BaseException as exc:
        try:
            report = capture.report(helper_sha256, lock_sha256)
            exc.image_encoder_internal_stages = report
            if callable(on_report):
                on_report(report)
        except Exception:
            pass
        raise
    finally:
        capture.close()
