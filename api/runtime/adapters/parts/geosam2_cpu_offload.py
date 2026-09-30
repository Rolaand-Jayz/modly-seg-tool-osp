"""Use GeoSAM2's native video/state CPU storage controls without source edits."""

from __future__ import annotations

from typing import Any, Callable


class GeoSAM2OffloadError(RuntimeError):
    pass


def install_cpu_offload(predictor: Any) -> tuple[Callable[[], None], dict[str, Any]]:
    """Wrap the already pinned/patched predictor ``init_state`` for one run.

    Call this after the video-index policy, and restore it before restoring that
    policy. The wrapper changes storage placement only; output parity on the
    RX 7900 GRE still requires a separate acceptance comparison.
    """
    previous = getattr(predictor, "init_state", None)
    if not callable(previous):
        raise GeoSAM2OffloadError("pinned GeoSAM2 predictor has no init_state method")
    namespace = getattr(predictor, "__dict__", {})
    had_instance = "init_state" in namespace
    instance_value = namespace.get("init_state")
    report: dict[str, Any] = {
        "policy": "pinned-geosam2-native-cpu-storage-v1",
        "requested_video_cpu_offload": True,
        "requested_state_cpu_offload": True,
        "state": "installed",
    }

    def init_state(*args: Any, **kwargs: Any) -> Any:
        for key in ("offload_video_to_cpu", "offload_state_to_cpu"):
            if key in kwargs and kwargs[key] is not True:
                raise GeoSAM2OffloadError(f"conflicting {key} argument")
            kwargs[key] = True
        state = previous(*args, **kwargs)
        if not isinstance(state, dict) or state.get("offload_video_to_cpu") is not True \
                or state.get("offload_state_to_cpu") is not True:
            report["state"] = "not_honored"
            raise GeoSAM2OffloadError("pinned GeoSAM2 did not honor both CPU offload controls")
        report["state"] = "active"
        return state

    predictor.init_state = init_state
    restored = False

    def restore() -> None:
        nonlocal restored
        if restored:
            return
        restored = True
        if had_instance:
            predictor.init_state = instance_value
        else:
            delattr(predictor, "init_state")

    return restore, report
