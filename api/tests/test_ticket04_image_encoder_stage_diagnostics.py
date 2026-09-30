from __future__ import annotations

import unittest

import numpy as np

from api.runtime.adapters.parts.geosam2_image_encoder_stage_diagnostics import (
    EXPECTED_BLOCK_COUNT,
    ImageEncoderStageCapture,
    run_with_image_encoder_stages,
)


class FakeHook:
    def __init__(self, callbacks: list) -> None:
        self.callbacks = callbacks

    def remove(self) -> None:
        self.callbacks.clear()


class FakeModule:
    def __init__(self, operation=lambda value: value) -> None:
        self.operation = operation
        self.pre_hooks = []
        self.hooks = []

    def register_forward_pre_hook(self, callback):
        self.pre_hooks.append(callback)
        return FakeHook(self.pre_hooks)

    def register_forward_hook(self, callback):
        self.hooks.append(callback)
        return FakeHook(self.hooks)

    def __call__(self, value):
        for callback in tuple(self.pre_hooks):
            callback(self, (value,))
        output = self.operation(value)
        for callback in tuple(self.hooks):
            callback(self, (value,), output)
        return output


class FakeTrunk:
    def __init__(self) -> None:
        self.patch_embed = FakeModule(lambda value: value + 1)
        self.blocks = [FakeModule(lambda value: value + 1)
                       for _ in range(EXPECTED_BLOCK_COUNT)]

    def _get_pos_embed(self):
        return np.array([2], dtype=np.float32)

    def __call__(self, value):
        value = self.patch_embed(value)
        value = value + self._get_pos_embed()
        for block in self.blocks:
            value = block(value)
        return value


class FakeImageEncoder(FakeModule):
    def __init__(self) -> None:
        super().__init__()
        self.trunk = FakeTrunk()

    def __call__(self, value):
        for callback in tuple(self.pre_hooks):
            callback(self, (value,))
        output = self.trunk(value)
        for callback in tuple(self.hooks):
            callback(self, (value,), output)
        return output


class FakeModel:
    def __init__(self) -> None:
        self.image_encoder = FakeImageEncoder()


class ImageEncoderStageDiagnosticsTests(unittest.TestCase):
    def test_capture_records_all_hiera_boundaries_for_each_role(self):
        model = FakeModel()
        capture = ImageEncoderStageCapture(model, torch_module=None)
        for _ in range(3):
            model.image_encoder(np.array([1], dtype=np.float32))
        report = capture.report("helper", "lock")
        capture.close()

        self.assertEqual(report["state"], "complete")
        self.assertEqual(report["captured_block_count"], EXPECTED_BLOCK_COUNT)
        self.assertEqual([call["role"] for call in report["roles"]], [
            "fresh_predictor", "reused_predictor_after_reset", "new_predictor_shared_model"])
        self.assertEqual(len(report["roles"][0]["events"]), EXPECTED_BLOCK_COUNT + 3)
        self.assertEqual(report["roles"][0]["events"][0]["stage"], "trunk.patch_embed.output")
        self.assertEqual(report["roles"][0]["events"][1]["stage"],
                         "trunk.positional_embedding.output")
        self.assertEqual(report["roles"][0]["events"][2]["stage"],
                         "trunk.blocks.0.input_after_position_add")
        self.assertNotIn("_get_pos_embed", model.image_encoder.trunk.__dict__)

    def test_lifecycle_wrapper_adds_report_and_restores_hooks(self):
        model = FakeModel()

        def lifecycle_runner(**_kwargs):
            for _ in range(3):
                model.image_encoder(np.array([1], dtype=np.float32))
            return {"acceptance_status": "not_assessed"}

        result = run_with_image_encoder_stages(
            lifecycle_runner=lifecycle_runner,
            helper_sha256="helper", lock_sha256="lock",
            model=model, torch_module=None)

        self.assertEqual(result["image_encoder_internal_stages"]["state"], "complete")
        self.assertEqual(model.image_encoder.pre_hooks, [])
        self.assertEqual(model.image_encoder.trunk.patch_embed.hooks, [])
        self.assertEqual(model.image_encoder.trunk.blocks[-1].hooks, [])
        self.assertNotIn("_get_pos_embed", model.image_encoder.trunk.__dict__)


if __name__ == "__main__":
    unittest.main()
