from __future__ import annotations

import ast
import json
import unittest
from types import FunctionType
from pathlib import Path
from unittest.mock import patch

from api.runtime.adapters.parts import geosam2
from api.runtime.adapters.parts.geosam2_video_index_policy import (
    INIT_STATE_AST_SHA256,
    POLICY_ID,
    PROPAGATE_AST_SHA256,
    VideoIndexPolicyError,
    _patch_init_state,
    _patch_propagation,
    _conditioned_seed_frame,
    _compile_patch,
    _preflight_and_remap_seed,
    _remap_seed_state,
    _source_view_count,
    install_video_index_policy,
    source_view_order,
)


class GeoSAM2VideoIndexPolicyTests(unittest.TestCase):
    @staticmethod
    def _locked_source_path():
        return (Path(__file__).resolve().parents[2]
                / ".modly-amd-runtime/source/geosam2-b5de23c60ab487d407b623d394a1614f9714761c"
                / "sam2/sam2_video_predictor_geosam2.py")

    @staticmethod
    def _function_at_source(path, name, body, globals_dict=None):
        namespace = {} if globals_dict is None else globals_dict
        code = compile(f"def {name}(*args, **kwargs):\n    {body}\n", str(path), "exec")
        exec(code, namespace)
        return namespace[name]

    def test_all_seed_views_rotate_once_and_keep_monotonic_temporal_slots(self):
        for seed in range(12):
            for reverse in (False, True):
                order = source_view_order(seed, 12, None, reverse)
                self.assertEqual(len(order), 12)
                self.assertEqual(order[0], seed)
                self.assertEqual(set(order), set(range(12)))
                self.assertEqual(len(set(order)), 12)
                self.assertEqual(order, [((seed + (-1 if reverse else 1) * i) % 12)
                                         for i in range(12)])

    def test_partial_limits_and_single_view(self):
        self.assertEqual(source_view_order(10, 12, 4, False), [10, 11, 0, 1])
        self.assertEqual(source_view_order(1, 3, None, True), [1, 0, 2])
        self.assertEqual(source_view_order(0, 1, None, False), [0])
        self.assertEqual(source_view_order(0, 12, 0, False), [])
        for args in ((-1, 12, None, False), (12, 12, None, False), (0, 0, None, False)):
            with self.assertRaises(VideoIndexPolicyError):
                source_view_order(*args)

    def test_indexable_tensor_like_source_frames_are_supported(self):
        class Frames:
            def __len__(self):
                return 12
            def __getitem__(self, index):
                if index < 0 or index >= 12:
                    raise IndexError(index)
                return f"view-{index}"
        self.assertEqual(_source_view_count(Frames()), 12)
        for value in (None, [], {}, "frames", object()):
            with self.subTest(value=type(value).__name__), self.assertRaises(VideoIndexPolicyError):
                _source_view_count(value)

    def test_omitted_start_uses_temporary_point_or_mask_conditioning_frame(self):
        for temporary_key in ("point_inputs_per_obj", "mask_inputs_per_obj"):
            state = {
                "output_dict_per_obj": {0: {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}},
                "temp_output_dict_per_obj": {0: {"cond_frame_outputs": {7: "prompt"},
                                                  "non_cond_frame_outputs": {}}},
            }
            self.assertEqual(_conditioned_seed_frame(state), 7, temporary_key)
        with self.assertRaises(VideoIndexPolicyError):
            _conditioned_seed_frame({"output_dict_per_obj": {}, "temp_output_dict_per_obj": {}})

    def test_omitted_start_uses_earliest_committed_or_temporary_conditioning_frame(self):
        state = {
            "output_dict_per_obj": {0: {"cond_frame_outputs": {9: "later-committed"}},
                                     1: {"cond_frame_outputs": {3: "earliest-committed"}}},
            "temp_output_dict_per_obj": {0: {"cond_frame_outputs": {5: "temporary"}}},
        }
        self.assertEqual(_conditioned_seed_frame(state), 3)

    def test_compile_patch_preserves_inference_mode_decorator(self):
        class TorchLike:
            @staticmethod
            def inference_mode():
                def decorate(function):
                    function.inference_mode_preserved = True
                    return function
                return decorate

        source = self._locked_source_path()
        original = self._function_at_source(
            source, "propagate_in_video_v2", "return None", {"torch": TorchLike})
        patched = _compile_patch(original, "propagate_in_video_v2", PROPAGATE_AST_SHA256,
                                 lambda node: None)
        self.assertTrue(getattr(patched, "inference_mode_preserved", False))

    def test_installed_wrapper_preflights_source_seed_before_remapping(self):
        source = self._locked_source_path()
        original_init = self._function_at_source(source, "init_state", "return None")
        original_propagate = self._function_at_source(
            source, "propagate_in_video_v2", "return None")
        calls = []

        def patched_init(self, *args, **kwargs):
            return None

        def patched_propagate(self, state, *, start_frame_idx,
                              max_frame_num_to_track=None, reverse=False):
            calls.append(("propagate", start_frame_idx,
                          state["point_inputs_per_obj"][0].copy()))
            return start_frame_idx

        def compile_locked(function, name, expected_digest, transform, globals_extra=None):
            return patched_init if name == "init_state" else patched_propagate

        class Predictor:
            init_state = original_init
            propagate_in_video_v2 = original_propagate

            def propagate_in_video_preflight(self, state):
                calls.append(("preflight", state["temp_output_dict_per_obj"][0]
                              ["cond_frame_outputs"].copy(),
                              state["point_inputs_per_obj"][0].copy()))
                # Model the upstream merge: encode at source frame 7 before
                # moving that output and the point input to temporal slot 0.
                value = state["temp_output_dict_per_obj"][0]["cond_frame_outputs"].pop(7)
                state["output_dict_per_obj"][0]["cond_frame_outputs"][7] = value

        predictor = Predictor()
        state = {
            "images": [object() for _ in range(12)],
            "num_frames": 12,
            "point_inputs_per_obj": {0: {7: "point-seed"}},
            "output_dict_per_obj": {0: {"cond_frame_outputs": {},
                                         "non_cond_frame_outputs": {}}},
            "temp_output_dict_per_obj": {0: {"cond_frame_outputs": {7: "encoded-seed"},
                                               "non_cond_frame_outputs": {}}},
        }
        with patch("api.runtime.adapters.parts.geosam2_video_index_policy._compile_patch",
                   side_effect=compile_locked):
            restore, _ = install_video_index_policy(predictor)
            try:
                result = predictor.propagate_in_video_v2(state)
            finally:
                restore()

        # The wrapped upstream call still receives source ID 7 so image
        # feature lookup remains correct; its patched loop maps that to slot 0.
        self.assertEqual(result, 7)
        self.assertEqual(calls[0], ("preflight", {7: "encoded-seed"}, {7: "point-seed"}))
        self.assertEqual(calls[1], ("propagate", 7, {0: "point-seed"}))
        self.assertEqual(state["output_dict_per_obj"][0]["cond_frame_outputs"],
                         {0: "encoded-seed"})
        self.assertEqual(state["temp_output_dict_per_obj"][0]["cond_frame_outputs"], {})

    def test_installed_wrapper_fails_closed_without_seed_before_preflight(self):
        source = self._locked_source_path()
        original_init = self._function_at_source(source, "init_state", "return None")
        original_propagate = self._function_at_source(
            source, "propagate_in_video_v2", "return None")
        calls = []

        def patched_init(self, *args, **kwargs):
            return None

        def patched_propagate(self, *args, **kwargs):
            calls.append("propagate")

        def compile_locked(function, name, expected_digest, transform, globals_extra=None):
            return patched_init if name == "init_state" else patched_propagate

        class Predictor:
            init_state = original_init
            propagate_in_video_v2 = original_propagate

            def propagate_in_video_preflight(self, state):
                calls.append("preflight")

        predictor = Predictor()
        state = {"images": [object() for _ in range(12)], "num_frames": 12,
                 "output_dict_per_obj": {}, "temp_output_dict_per_obj": {}}
        with patch("api.runtime.adapters.parts.geosam2_video_index_policy._compile_patch",
                   side_effect=compile_locked):
            restore, _ = install_video_index_policy(predictor)
            try:
                with self.assertRaises(VideoIndexPolicyError):
                    predictor.propagate_in_video_v2(state)
            finally:
                restore()
        self.assertEqual(calls, [])

    def test_seed_prompts_and_outputs_move_to_first_temporal_slot(self):
        state = {
            "point_inputs_per_obj": {0: {7: "points"}},
            "mask_inputs_per_obj": {0: {7: "mask"}},
            "frames_tracked_per_obj": {0: {7: "tracked"}},
            "output_dict_per_obj": {0: {"cond_frame_outputs": {7: "condition"},
                                         "non_cond_frame_outputs": {}}},
            "temp_output_dict_per_obj": {0: {"cond_frame_outputs": {7: "temp"},
                                               "non_cond_frame_outputs": {}}},
        }
        _remap_seed_state(state, 7)
        self.assertEqual(state["point_inputs_per_obj"][0], {0: "points"})
        self.assertEqual(state["mask_inputs_per_obj"][0], {0: "mask"})
        self.assertEqual(state["frames_tracked_per_obj"][0], {0: "tracked"})
        self.assertEqual(state["output_dict_per_obj"][0]["cond_frame_outputs"], {0: "condition"})
        self.assertEqual(state["temp_output_dict_per_obj"][0]["cond_frame_outputs"], {0: "temp"})
        reverse_state = {"point_inputs_per_obj": {0: {7: "points"}}}
        _remap_seed_state(reverse_state, 7, 11)
        self.assertEqual(reverse_state["point_inputs_per_obj"][0], {11: "points"})
        with self.assertRaises(VideoIndexPolicyError):
            _remap_seed_state({"point_inputs_per_obj": {0: {0: "collision", 7: "seed"}}}, 7)

    def test_preflight_encodes_original_source_frame_then_remaps_timeline(self):
        calls = []
        class Predictor:
            def propagate_in_video_preflight(self, state):
                calls.append(state["output_dict_per_obj"][0]["cond_frame_outputs"].copy())
                # Match upstream: temporary point output is merged under its source ID.
                state["output_dict_per_obj"][0]["cond_frame_outputs"][7] = "encoded-at-source-7"
                state["temp_output_dict_per_obj"][0]["cond_frame_outputs"].clear()

        state = {
            "point_inputs_per_obj": {0: {7: "prompt"}},
            "output_dict_per_obj": {0: {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}},
            "temp_output_dict_per_obj": {0: {"cond_frame_outputs": {7: "temporary"},
                                              "non_cond_frame_outputs": {}}},
        }
        _preflight_and_remap_seed(Predictor(), state, 7, 0)
        self.assertEqual(calls, [{}])
        self.assertEqual(state["output_dict_per_obj"][0]["cond_frame_outputs"],
                         {0: "encoded-at-source-7"})
        self.assertEqual(state["point_inputs_per_obj"][0], {0: "prompt"})
        self.assertEqual(state["temp_output_dict_per_obj"][0]["cond_frame_outputs"], {})

    def test_reused_predictor_state_must_be_reset_between_seed_views(self):
        # GeoSAM2's caller resets this state before every seed. The policy must
        # fail closed if a prior run's remapped outputs were left behind.
        state = {"point_inputs_per_obj": {0: {0: "prior", 7: "second-seed"}},
                 "output_dict_per_obj": {0: {"cond_frame_outputs": {0: "prior", 7: "second-seed"}}}}
        with self.assertRaises(VideoIndexPolicyError):
            _remap_seed_state(state, 7, 0)
        state["point_inputs_per_obj"].clear()
        state["output_dict_per_obj"].clear()
        state["point_inputs_per_obj"][0] = {7: "second-seed"}
        state["output_dict_per_obj"][0] = {"cond_frame_outputs": {7: "second-seed"}}
        _remap_seed_state(state, 7, 0)
        self.assertEqual(state["point_inputs_per_obj"][0], {0: "second-seed"})
        self.assertEqual(state["output_dict_per_obj"][0]["cond_frame_outputs"], {0: "second-seed"})

    def test_adapter_verifies_versioned_policy_lock(self):
        lock = Path(geosam2.__file__).with_name(geosam2.VIDEO_INDEX_POLICY_LOCK_NAME)
        identity = geosam2._verify_video_index_policy(lock)
        record = json.loads(lock.read_text())
        self.assertEqual(identity["policy_id"], POLICY_ID)
        self.assertEqual(record["module_sha256"], identity["module_sha256"])
        self.assertEqual(identity["schema"], "modly.ticket04.geosam2-video-index-policy.v1")

    def test_transforms_match_and_correct_the_pinned_source_methods(self):
        source_dir = Path(__file__).resolve().parents[2] / ".modly-amd-runtime/source/geosam2-b5de23c60ab487d407b623d394a1614f9714761c"
        path = source_dir / "sam2/sam2_video_predictor_geosam2.py"
        tree = ast.parse(path.read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == "SAM2VideoPredictor")
        methods = {node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)
                   and node.name in {"init_state", "propagate_in_video_v2"}}
        self.assertEqual(len(methods), 2)
        import hashlib
        self.assertEqual(hashlib.sha256(ast.unparse(methods["init_state"]).encode()).hexdigest(),
                         INIT_STATE_AST_SHA256)
        self.assertEqual(hashlib.sha256(ast.unparse(methods["propagate_in_video_v2"]).encode()).hexdigest(),
                         PROPAGATE_AST_SHA256)
        self.assertEqual([ast.unparse(item) for item in methods["propagate_in_video_v2"].decorator_list],
                         ["torch.inference_mode()"])
        _patch_init_state(methods["init_state"])
        _patch_propagation(methods["propagate_in_video_v2"])
        patched = ast.unparse(tree)
        self.assertIn("inference_state['num_frames'] = len(images)", patched)
        self.assertIn("_modly_source_view_order(start_frame_idx, num_frames, max_frame_num_to_track, reverse)",
                      patched)
        self.assertIn("frame_idx = _modly_temporal_slot(frame_idx_, num_frames, reverse)", patched)
        self.assertNotIn("processing_order = [start_frame_idx] + list(range(start_frame_idx, end_frame_idx))", patched)
        self.assertNotIn("% 13", patched)


if __name__ == "__main__":
    unittest.main()
