from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import torch
from evaluator import SUPPORTED_LABELS
import smolvlm2_candidate as subject


class FakeTokenizer:
    def __init__(self, overrides=None):
        self.overrides = overrides or {}

    def encode(self, text, add_special_tokens=False):
        return self.overrides.get(text, [ord(text)])


class FakeModel:
    training = False

    def __init__(self, logits):
        self.config = SimpleNamespace(vocab_size=128)
        self.logits = logits
        self.calls = 0

    def __call__(self, **kwargs):
        self.calls += 1
        assert kwargs["use_cache"] is False
        assert kwargs["return_dict"] is True
        return SimpleNamespace(logits=self.logits)


def token_ids():
    return {label: index + 20 for index, label in enumerate(SUPPORTED_LABELS)}


def model_inputs():
    return {"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
            "pixel_values": torch.zeros((1, 1, 3, 4, 4))}


class SmolVLM2CandidateTests(unittest.TestCase):
    def test_frozen_prompt_and_code_order_match_evaluator(self):
        self.assertEqual(tuple(label for _code, label, _display in subject.CODE_LABELS), SUPPORTED_LABELS)
        self.assertEqual(tuple(code for code, _label, _display in subject.CODE_LABELS), tuple("ABCDE"))
        self.assertEqual(len(subject.PROMPT_SHA256), 64)

    def test_answer_codes_are_distinct_single_tokens(self):
        mapping = subject.build_answer_token_ids(FakeTokenizer())
        self.assertEqual(tuple(mapping), SUPPORTED_LABELS)
        self.assertEqual(tuple(mapping.values()), tuple(ord(code) for code in "ABCDE"))
        with self.assertRaises(subject.SmolVLM2CandidateScoringError):
            subject.build_answer_token_ids(FakeTokenizer({"C": [3, 4]}))
        with self.assertRaises(subject.SmolVLM2CandidateScoringError):
            subject.build_answer_token_ids(FakeTokenizer({"E": [ord("A")]}))

    def test_returns_exact_five_ordered_raw_logits_at_final_prompt_position(self):
        logits = torch.zeros((1, 3, 128), dtype=torch.float32)
        expected = [1.25, -2.5, 3.75, 4.5, -5.0]
        for token_id, value in zip(token_ids().values(), expected):
            logits[0, 2, token_id] = value
        model = FakeModel(logits)
        scores = subject.score_raw_next_token_logits(model, model_inputs(), token_ids())
        self.assertEqual(tuple(scores), SUPPORTED_LABELS)
        self.assertEqual(tuple(scores.values()), tuple(expected))
        self.assertEqual(model.calls, 1)

    def test_rejects_wrong_label_mapping_order_or_set(self):
        ids = token_ids()
        reversed_ids = dict(reversed(tuple(ids.items())))
        logits = torch.zeros((1, 3, 128))
        with self.assertRaises(subject.SmolVLM2CandidateScoringError):
            subject.score_raw_next_token_logits(FakeModel(logits), model_inputs(), reversed_ids)
        with self.assertRaises(subject.SmolVLM2CandidateScoringError):
            subject.score_raw_next_token_logits(FakeModel(logits), model_inputs(), {k: v for k, v in ids.items() if k != SUPPORTED_LABELS[-1]})

    def test_rejects_malformed_or_misaligned_logits(self):
        ids = token_ids()
        cases = [torch.zeros((3, 128)), torch.zeros((2, 3, 128)), torch.zeros((1, 2, 128)), torch.zeros((1, 3, 127))]
        for logits in cases:
            with self.subTest(shape=tuple(logits.shape)), self.assertRaises(subject.SmolVLM2CandidateScoringError):
                subject.score_raw_next_token_logits(FakeModel(logits), model_inputs(), ids)

    def test_rejects_nonfinite_logits(self):
        ids = token_ids()
        logits = torch.zeros((1, 3, 128))
        logits[0, 2, ids[SUPPORTED_LABELS[2]]] = float("nan")
        with self.assertRaises(subject.SmolVLM2CandidateScoringError):
            subject.score_raw_next_token_logits(FakeModel(logits), model_inputs(), ids)

    def test_rejects_non_eval_model_and_masked_input(self):
        ids = token_ids()
        logits = torch.zeros((1, 3, 128))
        model = FakeModel(logits)
        model.training = True
        with self.assertRaises(subject.SmolVLM2CandidateScoringError):
            subject.score_raw_next_token_logits(model, model_inputs(), ids)
        masked = model_inputs()
        masked["attention_mask"][0, 1] = 0
        with self.assertRaises(subject.SmolVLM2CandidateScoringError):
            subject.score_raw_next_token_logits(FakeModel(logits), masked, ids)


if __name__ == "__main__":
    unittest.main()
