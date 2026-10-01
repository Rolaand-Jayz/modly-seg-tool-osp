from __future__ import annotations

import unittest

import numpy as np

from api.runtime.adapters.parts.geosam2_first_conv_diagnostics import (
    FirstConvCapture,
    FirstConvCaptureError,
    run_with_first_conv,
)


class FakeDevice:
    type = "cuda"

    def __str__(self) -> str:
        return "cuda:0"


class FakeTensor:
    def __init__(self, value):
        self.array = np.asarray(value, dtype=np.float32)
        self.shape = self.array.shape
        self.dtype = "torch.float32"
        self.device = FakeDevice()

    def detach(self):
        return self

    def numel(self):
        return self.array.size

    def element_size(self):
        return self.array.dtype.itemsize

    def is_floating_point(self):
        return True

    def is_complex(self):
        return False

    def to(self, **_kwargs):
        return self

    def contiguous(self):
        return self

    def view(self, _dtype):
        return FakeTensorBytes(self.array.view(np.uint8))


class FakeTensorBytes:
    def __init__(self, array):
        self.array = array

    def reshape(self, *shape):
        return FakeTensorBytes(self.array.reshape(*shape))

    def numpy(self):
        return self.array


class FakeHandle:
    def __init__(self, hooks, callback):
        self.hooks = hooks
        self.callback = callback

    def remove(self):
        if self.callback in self.hooks:
            self.hooks.remove(self.callback)


class FakeModule:
    def __init__(self, operation=lambda value: value):
        self.operation = operation
        self.pre_hooks = []
        self.hooks = []

    def register_forward_pre_hook(self, callback):
        self.pre_hooks.append(callback)
        return FakeHandle(self.pre_hooks, callback)

    def register_forward_hook(self, callback):
        self.hooks.append(callback)
        return FakeHandle(self.hooks, callback)

    def __call__(self, value):
        for callback in tuple(self.pre_hooks):
            callback(self, (value,))
        output = self.operation(value)
        for callback in tuple(self.hooks):
            callback(self, (value,), output)
        return output


class Conv2d(FakeModule):
    def __init__(self):
        super().__init__(lambda value: FakeTensor(np.broadcast_to(
            np.mean(value.array, axis=1, keepdims=True) + 2,
            (value.array.shape[0], 112, value.array.shape[2], value.array.shape[3]))))
        self.in_channels = 3
        self.out_channels = 112
        self.kernel_size = (7, 7)
        self.stride = (4, 4)
        self.padding = (3, 3)
        self.groups = 1
        self.weight = FakeTensor(np.ones((112, 3, 7, 7), dtype=np.float32))
        self.bias = FakeTensor(np.zeros((112,), dtype=np.float32))


class FakePatchEmbed(FakeModule):
    def __init__(self):
        super().__init__()
        self.proj = Conv2d()

    def __call__(self, value):
        for callback in tuple(self.pre_hooks):
            callback(self, (value,))
        output = self.proj(value)
        return output


class FakeEncoder(FakeModule):
    def __init__(self):
        super().__init__()
        self.trunk = type("Trunk", (), {})()
        self.trunk.patch_embed = FakePatchEmbed()

    def __call__(self, value):
        for callback in tuple(self.pre_hooks):
            callback(self, (value,))
        return self.trunk.patch_embed(value)


class FakeModel:
    def __init__(self):
        self.image_encoder = FakeEncoder()


class FakeTorch:
    uint8 = np.uint8

    @staticmethod
    def is_autocast_enabled(device_type):
        return device_type == "cuda"

    @staticmethod
    def get_autocast_dtype(device_type):
        return "torch.bfloat16" if device_type == "cuda" else "torch.float32"

    @staticmethod
    def are_deterministic_algorithms_enabled():
        return False

    @staticmethod
    def is_deterministic_algorithms_warn_only_enabled():
        return False

    class backends:
        class cuda:
            class matmul:
                allow_tf32 = True

        class cudnn:
            allow_tf32 = True
            deterministic = False


class FirstConvDiagnosticTests(unittest.TestCase):
    def _execute(self):
        model = FakeModel()

        def runner(**_kwargs):
            for value in (1, 2, 3):
                model.image_encoder(FakeTensor(np.full((1, 3, 2, 2), value)))
            return {"state": "complete"}

        result = run_with_first_conv(
            lifecycle_runner=runner, helper_sha256="helper", lock_sha256="lock",
            model=model, torch_module=FakeTorch())
        return model, result["image_encoder_first_conv"]

    def test_records_patch_and_exact_first_conv_boundaries_per_lifecycle_call(self):
        model, report = self._execute()
        self.assertEqual(report["schema"], "modly.ticket04.geosam2-hiera-first-conv-diagnostics/1")
        self.assertEqual(report["state"], "complete")
        self.assertFalse(report["settings_mutated"])
        self.assertFalse(report["payloads_persisted"])
        self.assertEqual([item["role"] for item in report["roles"]], [
            "fresh_predictor", "reused_predictor_after_reset", "new_predictor_shared_model"])
        for item in report["roles"]:
            self.assertEqual(item["state"], "complete")
            self.assertEqual(item["patch_embed_input"]["shape"], [1, 3, 2, 2])
            self.assertEqual(item["first_conv_input"]["shape"], [1, 3, 2, 2])
            self.assertNotEqual(item["first_conv_output"]["sha256"], item["first_conv_input"]["sha256"])
            self.assertEqual(item["first_conv_output"]["shape"], [1, 112, 2, 2])
            self.assertEqual(item["weight"]["shape"], [112, 3, 7, 7])
            self.assertEqual(item["bias"]["shape"], [112])
            self.assertEqual(item["runtime_state"]["autocast_enabled_for_input_device"], True)
            self.assertEqual(item["runtime_state"]["autocast_dtype_for_input_device"], "torch.bfloat16")
            self.assertEqual(item["runtime_state"]["cuda_matmul_allow_tf32"], True)
        self.assertEqual(len({item["weight"]["sha256"] for item in report["roles"]}), 1)
        self.assertEqual(len({item["bias"]["sha256"] for item in report["roles"]}), 1)
        self.assertEqual(model.image_encoder.pre_hooks, [])
        self.assertEqual(model.image_encoder.trunk.patch_embed.pre_hooks, [])
        self.assertEqual(model.image_encoder.trunk.patch_embed.proj.pre_hooks, [])
        self.assertEqual(model.image_encoder.trunk.patch_embed.proj.hooks, [])

    def test_rejects_unpinned_conv_target(self):
        model = FakeModel()
        model.image_encoder.trunk.patch_embed.proj = type("OtherConv", (Conv2d,), {})()
        with self.assertRaises(FirstConvCaptureError):
            FirstConvCapture(model, FakeTorch())


if __name__ == "__main__":
    unittest.main()
