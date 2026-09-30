from __future__ import annotations

import unittest
import sys
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from runtime.adapters.parts.sonata_ops import segment_csr, submanifold_conv3d
from runtime.adapters.parts.sonata_spconv_compat import (
    SparseConvTensor,
    SubMConv3d,
    install_spconv_compat,
)
from runtime.adapters.parts.sonata_scatter_compat import install_torch_scatter_compat
from runtime.adapters.parts.sonata_dict_compat import SonataDictCompat, install_sonata_dict_compat
from runtime.adapters.parts.sonata_timm_compat import SonataDropPathCompat, install_sonata_timm_compat
from runtime.adapters.parts.p3sam import (
    P3SAM_SOURCE_REVISION,
    SONATA_CONFIG_OVERRIDE_SHA256,
    SONATA_UPSTREAM_CONFIG_SHA256,
    _import_p3sam_modules,
    install_p3sam_compatibility_shims,
)
from runtime.adapters.parts.sonata_transform_overlay import (
    SONATA_TRANSFORM_OVERLAY_SHA256,
    SONATA_TRANSFORM_SOURCE_SHA256,
    render_sonata_transform_overlay,
)
from runtime.adapters.parts.p3sam_auto_mask_overlay import (
    P3SAM_AUTO_MASK_OVERLAY_SHA256,
    P3SAM_AUTO_MASK_SOURCE_SHA256,
    render_p3sam_auto_mask_overlay,
)
from runtime.adapters.parts.probe import _native_inventory


class SonataPortableOpsTests(unittest.TestCase):
    def test_submanifold_conv_matches_dense_conv3d_on_sparse_grid(self) -> None:
        torch.manual_seed(104)
        spatial_shape = (4, 5, 3)
        coordinates = torch.tensor(
            [[0, 0, 0], [0, 1, 2], [1, 1, 1], [2, 3, 1], [3, 4, 2]],
            dtype=torch.int64,
        )
        features = torch.randn((len(coordinates), 2), dtype=torch.float64)
        weight = torch.randn((3, 2, 3, 3, 3), dtype=torch.float64)
        bias = torch.randn((3,), dtype=torch.float64)
        dense_input = features.new_zeros((1, 2, *spatial_shape))
        dense_input[0, :, coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]] = features.T

        expected_dense = F.conv3d(dense_input, weight, bias, padding=1)
        expected = expected_dense[0, :, coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]].T
        actual = submanifold_conv3d(
            features,
            coordinates,
            weight,
            bias,
            spatial_shape=spatial_shape,
        )

        self.assertEqual(actual.shape, expected.shape)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_submanifold_conv_handles_kernel_and_absent_bias(self) -> None:
        coordinates = torch.tensor([[1, 1, 1], [1, 1, 2], [2, 2, 2]], dtype=torch.int32)
        features = torch.tensor([[1.0], [2.0], [-1.0]], dtype=torch.float32)
        weight = torch.arange(27, dtype=torch.float32).reshape(1, 1, 3, 3, 3)
        dense_input = torch.zeros((1, 1, 4, 4, 4))
        dense_input[0, 0, coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]] = features[:, 0]
        expected_dense = F.conv3d(dense_input, weight, padding=1)
        expected = expected_dense[0, 0, coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]]
        actual = submanifold_conv3d(features, coordinates, weight, spatial_shape=(4, 4, 4))[:, 0]
        torch.testing.assert_close(actual, expected)

    def test_submanifold_conv_rejects_duplicate_or_out_of_range_coordinates(self) -> None:
        features = torch.ones((2, 1))
        weight = torch.ones((1, 1, 1, 1, 1))
        with self.assertRaisesRegex(ValueError, "unique"):
            submanifold_conv3d(features, torch.tensor([[0, 0, 0], [0, 0, 0]]), weight, spatial_shape=(1, 1, 1))
        with self.assertRaisesRegex(ValueError, "outside"):
            submanifold_conv3d(features, torch.tensor([[0, 0, 0], [1, 0, 0]]), weight, spatial_shape=(1, 1, 1))

    def test_segment_csr_matches_expected_reductions(self) -> None:
        values = torch.tensor([[1.0, 5.0], [3.0, 1.0], [2.0, 7.0], [-2.0, 4.0], [8.0, 0.0]])
        indptr = torch.tensor([0, 2, 5], dtype=torch.int64)
        expected = {
            "sum": torch.tensor([[4.0, 6.0], [8.0, 11.0]]),
            "mean": torch.tensor([[2.0, 3.0], [8.0 / 3.0, 11.0 / 3.0]]),
            "min": torch.tensor([[1.0, 1.0], [-2.0, 0.0]]),
            "max": torch.tensor([[3.0, 5.0], [8.0, 7.0]]),
        }
        for reduction, target in expected.items():
            with self.subTest(reduction=reduction):
                torch.testing.assert_close(segment_csr(values, indptr, reduce=reduction), target)

    def test_spconv_subset_krsc_weights_and_batch_column_match_dense_conv3d(self) -> None:
        torch.manual_seed(804)
        spatial_shape = (5, 4, 6)
        indices = torch.tensor(
            [
                [0, 0, 0, 0], [0, 1, 1, 2], [0, 3, 2, 4],
                [1, 0, 3, 5], [1, 2, 1, 1], [1, 4, 3, 0],
            ], dtype=torch.int32
        )
        features = torch.randn((len(indices), 2), dtype=torch.float64)
        sparse = SparseConvTensor(features, indices, spatial_shape, batch_size=2)
        layer = SubMConv3d(2, 3, kernel_size=3, stride=1, padding=1, bias=True).double()
        with torch.no_grad():
            layer.weight.copy_(torch.randn_like(layer.weight))  # spconv KRSC: [out,kD,kH,kW,in]
            layer.bias.copy_(torch.randn_like(layer.bias))

        actual = layer(sparse)
        dense_expected = F.conv3d(
            sparse.dense(),
            layer.weight.permute(0, 4, 1, 2, 3).contiguous(),
            layer.bias,
            stride=1,
            padding=1,
        )
        expected = dense_expected[
            indices[:, 0].long(), :, indices[:, 1].long(), indices[:, 2].long(), indices[:, 3].long()
        ]
        self.assertTrue(torch.equal(actual.indices, sparse.indices))
        torch.testing.assert_close(actual.features, expected, rtol=1e-12, atol=1e-12)

    def test_spconv_subset_padding_zero_and_stride_contract(self) -> None:
        indices = torch.tensor([[0, 1, 1, 1], [0, 2, 2, 2], [0, 1, 2, 1]], dtype=torch.int32)
        features = torch.tensor([[1.0], [2.0], [-1.0]])
        sparse = SparseConvTensor(features, indices, (5, 5, 5), batch_size=1)
        # Pinned Sonata constructs `SubMConv3d(channels, channels, 3, bias=True)`
        # without padding, so the compatibility default follows upstream 0.
        layer = SubMConv3d(1, 1, 3, bias=False)
        with torch.no_grad():
            layer.weight.copy_(torch.arange(27, dtype=torch.float32).reshape(1, 3, 3, 3, 1))
        actual = layer(sparse)
        dense_expected = F.conv3d(sparse.dense(), layer.weight.permute(0, 4, 1, 2, 3).contiguous(), padding=0)
        expected = dense_expected[
            indices[:, 0].long(), :, indices[:, 1].long(), indices[:, 2].long(), indices[:, 3].long()
        ]
        torch.testing.assert_close(actual.features, expected)

        with self.assertRaisesRegex(ValueError, "stride one"):
            SubMConv3d(1, 1, 3, stride=2)

    def test_compat_import_namespace_and_pointsequential_module_predicate(self) -> None:
        if "spconv" in sys.modules or "spconv.pytorch" in sys.modules:
            self.skipTest("a spconv module is already loaded in this interpreter")
        pytorch_namespace = install_spconv_compat()
        try:
            import spconv
            import spconv.pytorch as imported_spconv

            self.assertIs(imported_spconv, pytorch_namespace)
            self.assertIs(imported_spconv.SparseConvTensor, SparseConvTensor)
            self.assertTrue(spconv.modules.is_spconv_module(SubMConv3d(1, 1, 1)))
            self.assertFalse(spconv.modules.is_spconv_module(torch.nn.Linear(1, 1)))
        finally:
            for name in ("spconv.pytorch", "spconv.modules", "spconv"):
                sys.modules.pop(name, None)

    def test_torch_scatter_namespace_routes_to_csr_implementation(self) -> None:
        if "torch_scatter" in sys.modules:
            self.skipTest("torch_scatter is already loaded in this interpreter")
        module = install_torch_scatter_compat()
        try:
            imported = importlib.import_module("torch_scatter")
            self.assertIs(imported, module)
            values = torch.tensor([[2.0, 4.0], [4.0, 8.0], [3.0, 9.0]])
            indptr = torch.tensor([0, 2, 3], dtype=torch.int64)
            result = imported.segment_csr(values, indptr, reduce="mean")
            torch.testing.assert_close(result, torch.tensor([[3.0, 6.0], [3.0, 9.0]]))
            self.assertTrue(imported._modly_sonata_compat)
        finally:
            sys.modules.pop("torch_scatter", None)

    def test_upstream_imports_are_requested_only_after_both_shims_install(self) -> None:
        if any(name in sys.modules for name in ("spconv", "spconv.pytorch", "torch_scatter", "addict", "timm")):
            self.skipTest("a native or adapter sparse module is already loaded in this interpreter")
        original_path = list(sys.path)
        observed: list[str] = []
        overlay_installed: list[Path] = []
        auto_mask_overlay_installed: list[Path] = []

        def overlay_installer(root: Path):
            overlay_installed.append(root)
            return {"provider": "modly.adapter.sonata_transform_overlay", "source_sha256": SONATA_TRANSFORM_SOURCE_SHA256}

        def auto_mask_overlay_installer(root: Path):
            auto_mask_overlay_installed.append(root)
            return {"provider": "modly.adapter.p3sam_auto_mask_overlay", "source_sha256": P3SAM_AUTO_MASK_SOURCE_SHA256}

        def loader(name: str):
            observed.append(name)
            if name == "torch":
                return torch
            if name in ("models.sonata", "auto_mask"):
                self.assertTrue(sys.modules["spconv.pytorch"]._modly_sonata_compat)
                self.assertTrue(sys.modules["torch_scatter"]._modly_sonata_compat)
                self.assertTrue(sys.modules["addict"]._modly_sonata_dict_compat)
                self.assertTrue(sys.modules["timm"]._modly_sonata_timm_compat)
                self.assertEqual(overlay_installed, [Path("/pinned/test-source")])
            if name == "auto_mask":
                self.assertEqual(auto_mask_overlay_installed, [Path("/pinned/test-source")])
            if name == "models.sonata":
                return SimpleNamespace(load=lambda *args, **kwargs: None)
            if name == "auto_mask":
                return SimpleNamespace(AutoMask=object, set_seed=lambda seed: None)
            raise AssertionError(f"unexpected import {name}")

        try:
            loaded = _import_p3sam_modules(
                Path("/pinned/test-source"),
                module_importer=loader,
                source_overlay_installer=overlay_installer,
                auto_mask_overlay_installer=auto_mask_overlay_installer,
            )
            self.assertEqual(observed, ["torch", "models.sonata", "auto_mask"])
            self.assertEqual(loaded[4]["spconv"]["provider"], "modly.adapter.sonata_spconv_compat")
            self.assertEqual(loaded[4]["torch_scatter"]["apis"], ["segment_csr"])
            self.assertEqual(loaded[4]["sonata_dict"]["provider"], "modly.adapter.sonata_dict_compat")
            self.assertEqual(loaded[4]["sonata_dict"]["import_namespace"], "addict.Dict")
            self.assertEqual(loaded[4]["timm_drop_path"]["provider"], "modly.adapter.sonata_timm_compat")
            self.assertEqual(loaded[4]["sonata_transform"]["provider"], "modly.adapter.sonata_transform_overlay")
            self.assertEqual(loaded[4]["p3sam_auto_mask"]["provider"], "modly.adapter.p3sam_auto_mask_overlay")
        finally:
            sys.path[:] = original_path
            for name in ("spconv.pytorch", "spconv.modules", "spconv", "torch_scatter", "addict", "timm.layers", "timm"):
                sys.modules.pop(name, None)

    def test_sonata_drop_path_matches_pinned_train_and_eval_behavior(self) -> None:
        values = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        module = SonataDropPathCompat(drop_prob=0.5)
        module.eval()
        rng_before = torch.random.get_rng_state()
        self.assertIs(module(values), values)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_before))

        module.train()
        torch.manual_seed(490)
        actual = module(values)
        torch.manual_seed(490)
        keep = 1.0 - 0.5
        mask = values.new_empty((values.shape[0],) + (1,) * (values.ndim - 1)).bernoulli_(keep)
        mask.div_(keep)
        expected = values * mask
        torch.testing.assert_close(actual, expected)
        self.assertTrue(torch.all((actual == 0) | (actual == values * 2)))

        all_drop = SonataDropPathCompat(drop_prob=1.0).train()
        self.assertTrue(torch.equal(all_drop(values), torch.zeros_like(values)))
        self.assertEqual(module.extra_repr(), "drop_prob=0.500")

    def test_sonata_timm_import_namespace_is_adapter_local_and_limited(self) -> None:
        if "timm" in sys.modules or "timm.layers" in sys.modules:
            self.skipTest("a timm module is already loaded in this interpreter")
        parent, layers = install_sonata_timm_compat()
        try:
            imported = importlib.import_module("timm.layers")
            self.assertIs(imported, layers)
            self.assertIs(parent.layers, layers)
            self.assertIs(imported.DropPath, SonataDropPathCompat)
            self.assertTrue(imported._modly_sonata_timm_compat)
            self.assertIn("not upstream timm", imported.__doc__)
        finally:
            sys.modules.pop("timm.layers", None)
            sys.modules.pop("timm", None)

    def test_sonata_dict_compat_matches_observed_point_mapping_semantics(self) -> None:
        nested = SonataDictCompat(
            feat=torch.tensor([[1.0]]),
            child={"enabled": True},
            records=[{"tag": "root"}],
            tupled=({"index": 2},),
        )
        self.assertIsInstance(nested, dict)
        self.assertIsInstance(nested.child, SonataDictCompat)
        self.assertTrue(nested.child.enabled)
        self.assertEqual(nested["child"]["enabled"], True)
        self.assertEqual(nested.records[0].tag, "root")
        self.assertEqual(nested.tupled[0].index, 2)
        nested.extra = "value"
        self.assertEqual(nested["extra"], "value")
        nested["extra"] = "updated"
        self.assertEqual(nested.extra, "updated")
        del nested.extra
        with self.assertRaises(AttributeError):
            _ = nested.extra
        with self.assertRaises(AttributeError):
            _ = nested.missing

    def test_sonata_dict_import_namespace_is_local_and_limited(self) -> None:
        if "addict" in sys.modules:
            self.skipTest("an addict module is already loaded in this interpreter")
        module = install_sonata_dict_compat()
        try:
            imported = importlib.import_module("addict")
            self.assertIs(imported, module)
            self.assertIs(imported.Dict, SonataDictCompat)
            self.assertTrue(imported._modly_sonata_dict_compat)
            self.assertIn("not upstream addict", imported.__doc__)
        finally:
            sys.modules.pop("addict", None)

    def test_sonata_override_identity_is_recorded_and_targets_only_flash_flag(self) -> None:
        override_path = Path(__file__).resolve().parents[1] / "runtime/adapters/parts/SONATA_CONFIG_OVERRIDE.json"
        payload = override_path.read_bytes()
        document = json.loads(payload)
        self.assertEqual(hashlib.sha256(payload).hexdigest(), SONATA_CONFIG_OVERRIDE_SHA256)
        self.assertEqual(document["upstream"]["source_revision"], P3SAM_SOURCE_REVISION)
        self.assertEqual(document["upstream"]["config_sha256"], SONATA_UPSTREAM_CONFIG_SHA256)
        self.assertEqual(document["overrides"], {"enable_flash": False})

    def test_sonata_transform_overlay_is_pinned_and_changes_only_lazy_imports(self) -> None:
        source_root = Path(os.environ.get("MODLY_P3SAM_SOURCE", ""))
        if not source_root.is_dir():
            self.skipTest("pinned P3-SAM source is not mounted for source-overlay identity test")
        original = (source_root / "XPart/partgen/models/sonata/transform.py").read_bytes()
        patched, identity = render_sonata_transform_overlay(source_root)
        self.assertEqual(hashlib.sha256(original).hexdigest(), SONATA_TRANSFORM_SOURCE_SHA256)
        self.assertEqual(identity["manifest_sha256"], SONATA_TRANSFORM_OVERLAY_SHA256)
        self.assertNotEqual(hashlib.sha256(original).hexdigest(), hashlib.sha256(patched).hexdigest())
        source = original.decode("utf-8")
        transformed = patched.decode("utf-8")
        self.assertEqual(source.count("import scipy\nimport scipy.ndimage\nimport scipy.interpolate\nimport scipy.stats\n"), 1)
        self.assertNotIn("\nimport scipy\n", transformed)
        self.assertIn("SCIPY_REQUIRED_FOR_ELASTIC_DISTORTION", transformed)
        self.assertEqual(
            transformed.count("scipy.ndimage.filters.convolve("),
            source.count("scipy.ndimage.filters.convolve("),
        )
        self.assertEqual(
            transformed.count("scipy.interpolate.RegularGridInterpolator("),
            source.count("scipy.interpolate.RegularGridInterpolator("),
        )
        relocated_imports = (
            "        try:\n"
            "            import scipy\n"
            "            import scipy.ndimage\n"
            "            import scipy.interpolate\n"
            "            import scipy.stats\n"
            "        except ImportError as exc:\n"
            "            raise RuntimeError(\n"
            "                \"SCIPY_REQUIRED_FOR_ELASTIC_DISTORTION: install the reviewed SciPy runtime dependency to use this training/augmentation transform\"\n"
            "            ) from exc\n"
        )
        marker = "        magnitude: noise multiplier\n        \"\"\"\n"
        self.assertIn(marker, transformed)
        global_import_marker = "import numbers\nimport numpy as np\n"
        self.assertEqual(transformed.count(global_import_marker), 1)
        restored = transformed.replace(relocated_imports, "", 1).replace(
            global_import_marker,
            "import numbers\nimport scipy\nimport scipy.ndimage\nimport scipy.interpolate\nimport scipy.stats\nimport numpy as np\n",
            1,
        )
        self.assertEqual(restored.encode("utf-8"), original)

    def test_pinned_default_transform_runs_without_scipy_and_explicit_elastic_fails_clearly(self) -> None:
        source_root = Path(os.environ.get("MODLY_P3SAM_SOURCE", ""))
        if not source_root.is_dir():
            self.skipTest("pinned P3-SAM source is not mounted for integration smoke")
        script = r'''import importlib.util, json, os, sys
from pathlib import Path
import numpy as np
from runtime.adapters.parts.p3sam import install_p3sam_compatibility_shims
from runtime.adapters.parts.sonata_transform_overlay import install_sonata_transform_overlay
root = Path(os.environ["MODLY_P3SAM_SOURCE"])
sys.path[:0] = [str(root / "P3-SAM" / "demo"), str(root / "P3-SAM"), str(root / "XPart" / "partgen")]
assert importlib.util.find_spec("scipy") is None
shims = install_p3sam_compatibility_shims()
overlay = install_sonata_transform_overlay(root)
shims["sonata_transform"] = overlay
from models import sonata
transform = sonata.transform
pipeline = transform.default()
names = [type(op).__name__ for op in pipeline.transforms]
assert names == ["CenterShift", "GridSample", "NormalizeColor", "ToTensor", "Collect"], names
assert "ElasticDistortion" not in names
points = np.asarray([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.01]], dtype=np.float32)
normals = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
np.random.seed(1234)
result = pipeline({"coord": points.copy(), "normal": normals.copy(), "color": np.ones_like(points), "batch": np.zeros(3, dtype=np.int64)})
inverse = result["inverse"].long()
centered = points - np.asarray([0.005, 0.005, 0.0], dtype=np.float32)
assert tuple(result["coord"].shape) == (3, 3)
assert tuple(result["feat"].shape) == (3, 9)
assert np.allclose(result["coord"][inverse].numpy(), centered, atol=1e-6)
assert np.allclose(result["color"][inverse].numpy(), np.ones_like(points) / 255.0, atol=1e-7)
assert result["offset"].tolist() == [3]
try:
    transform.ElasticDistortion.elastic_distortion(points.copy(), 0.2, 0.4)
except RuntimeError as exc:
    assert "SCIPY_REQUIRED_FOR_ELASTIC_DISTORTION" in str(exc), str(exc)
else:
    raise AssertionError("ElasticDistortion unexpectedly ran without SciPy")
assert overlay["source_sha256"] and overlay["effective_source_sha256"]
print(json.dumps({"default_transforms": names, "output_keys": sorted(result), "overlay": overlay, "shim_keys": sorted(shims)}))
'''
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(filter(None, (env.get("PYTHONPATH"), str(Path(__file__).resolve().parents[1]))))
        completed = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        report = json.loads(completed.stdout.strip().splitlines()[-1])
        self.assertEqual(report["default_transforms"], ["CenterShift", "GridSample", "NormalizeColor", "ToTensor", "Collect"])
        self.assertIn("sonata_transform", report["shim_keys"])

    def test_p3sam_auto_mask_overlay_defers_debug_pca_and_routes_native_ops(self) -> None:
        source_root = Path(os.environ.get("MODLY_P3SAM_SOURCE", ""))
        if not source_root.is_dir():
            self.skipTest("pinned P3-SAM source is not mounted for auto_mask overlay identity test")
        original = (source_root / "P3-SAM/demo/auto_mask.py").read_bytes()
        patched, identity = render_p3sam_auto_mask_overlay(source_root)
        self.assertEqual(hashlib.sha256(original).hexdigest(), P3SAM_AUTO_MASK_SOURCE_SHA256)
        self.assertEqual(identity["manifest_sha256"], P3SAM_AUTO_MASK_OVERLAY_SHA256)
        self.assertFalse(identity["debug_pca_required_by_selected_path"])
        source = original.decode("utf-8")
        transformed = patched.decode("utf-8")
        self.assertEqual(source.count("from sklearn.decomposition import PCA\n"), 1)
        self.assertNotIn("\nfrom sklearn.decomposition import PCA\n", transformed)
        self.assertEqual(source.count("pca = PCA(n_components=3)"), 1)
        self.assertEqual(transformed.count("pca = PCA(n_components=3)"), 1)
        self.assertIn("if save_mid_res:", transformed)
        self.assertIn("SKLEARN_REQUIRED_FOR_P3SAM_DEBUG_PCA", transformed)
        self.assertNotIn("import numba", transformed)
        self.assertIn(
            "from runtime.adapters.parts.p3sam_face_adjacency import build_adjacent_faces_numba",
            transformed,
        )
        marker = "import trimesh\nfrom runtime.adapters.parts import p3sam_fpsample as fpsample\n"
        self.assertEqual(transformed.count(marker), 1)

    def test_installed_shim_output_wiring_preserves_feature_rows_and_topology(self) -> None:
        if any(name in sys.modules for name in ("spconv", "spconv.pytorch", "torch_scatter")):
            self.skipTest("a native or adapter sparse module is already loaded in this interpreter")
        original_path = list(sys.path)
        try:
            metadata = install_p3sam_compatibility_shims()
            spconv = importlib.import_module("spconv.pytorch")
            scatter = importlib.import_module("torch_scatter")
            features = torch.tensor([[1.0], [2.0]])
            indices = torch.tensor([[0, 1, 1, 1], [0, 2, 2, 2]], dtype=torch.int32)
            sparse = spconv.SparseConvTensor(features, indices, (4, 4, 4), 1)
            layer = spconv.SubMConv3d(1, 2, 1, bias=True)
            output = layer(sparse)
            self.assertTrue(torch.equal(output.indices, indices))
            self.assertEqual(output.features.shape, (2, 2))
            pooled = scatter.segment_csr(torch.tensor([[1.0], [3.0]]), torch.tensor([0, 2]), reduce="mean")
            self.assertEqual(pooled.item(), 2.0)
            self.assertEqual(set(metadata), {"spconv", "torch_scatter", "sonata_dict", "timm_drop_path"})
            inventory = _native_inventory()
            self.assertEqual(inventory["python_modules"]["spconv"]["provider"], "in_tree_adapter_shim")
            self.assertEqual(inventory["python_modules"]["torch_scatter"]["provider"], "in_tree_adapter_shim")
            self.assertEqual(inventory["python_modules"]["timm"]["provider"], "in_tree_adapter_shim")
        finally:
            sys.path[:] = original_path
            for name in ("spconv.pytorch", "spconv.modules", "spconv", "torch_scatter", "addict", "timm.layers", "timm"):
                sys.modules.pop(name, None)


if __name__ == "__main__":
    unittest.main()
