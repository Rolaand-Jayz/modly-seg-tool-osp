from __future__ import annotations

from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

import torch

IDENTITY_DIR = Path(__file__).resolve().parents[1] / "runtime" / "adapters" / "material-identity"
sys.path.insert(0, str(IDENTITY_DIR))

from evaluator import SUPPORTED_LABELS
from qwen3_vl_scoring import CandidateScoringError, score_candidate_continuations


class FixedLogitModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 16, *, trim_logits: bool = False, nonfinite: bool = False):
        super().__init__()
        self.config = SimpleNamespace(vocab_size=vocab_size)
        self.trim_logits = trim_logits
        self.nonfinite = nonfinite
        self.seen_inputs: list[torch.Tensor] = []

    def forward(self, input_ids, attention_mask=None, use_cache=True, return_dict=True, **kwargs):
        self.seen_inputs.append(input_ids.detach().clone())
        vocab = torch.arange(self.config.vocab_size, dtype=torch.float32, device=input_ids.device)
        logits = vocab.view(1, 1, -1).expand(input_ids.shape[0], input_ids.shape[1], -1).clone()
        if self.nonfinite:
            logits[0, -1, 0] = torch.inf
        if self.trim_logits:
            logits = logits[:, :-1, :]
        return SimpleNamespace(logits=logits)


def candidate_tokens():
    return {
        "rubber_latex": [4],
        "glass": [5, 6],
        "clear_plastic": [7],
        "paint_plaster_enamel": [8, 9, 10],
        "metal": [11],
    }


def prompt(values=(1, 2, 3)):
    return {
        "input_ids": torch.tensor([values], dtype=torch.long),
        "attention_mask": torch.ones((1, len(values)), dtype=torch.long),
    }


class Ticket07QwenScoringTests(unittest.TestCase):
    def setUp(self):
        self.model = FixedLogitModel().eval()
        self.candidates = candidate_tokens()

    def test_returns_exact_five_evaluator_raw_logit_keys_in_order(self):
        scores = score_candidate_continuations(self.model, prompt(), self.candidates)
        self.assertEqual(tuple(scores), SUPPORTED_LABELS)
        self.assertEqual(len(scores), 5)
        self.assertTrue(all(isinstance(value, float) for value in scores.values()))

    def test_scores_only_continuation_prediction_positions(self):
        scores = score_candidate_continuations(self.model, prompt(), self.candidates)
        log_probs = torch.log_softmax(torch.arange(16, dtype=torch.float32), dim=0)
        for label, tokens in self.candidates.items():
            expected = float(sum(log_probs[token].item() for token in tokens))
            self.assertAlmostEqual(scores[label], expected, places=6)
        self.assertTrue(all(seen.shape[1] == 3 + len(self.candidates[label])
                            for seen, label in zip(self.model.seen_inputs, SUPPORTED_LABELS)))

    def test_prompt_token_positions_do_not_contribute_to_score(self):
        first = score_candidate_continuations(self.model, prompt((1, 2, 3)), self.candidates)
        second = score_candidate_continuations(self.model, prompt((12, 13, 14)), self.candidates)
        self.assertEqual(first, second)

    def test_repeated_scoring_is_deterministic(self):
        first = score_candidate_continuations(self.model, prompt(), self.candidates)
        second = score_candidate_continuations(self.model, prompt(), self.candidates)
        self.assertEqual(first, second)

    def test_rejects_missing_or_extra_label(self):
        missing = dict(self.candidates)
        missing.pop("metal")
        with self.assertRaises(CandidateScoringError):
            score_candidate_continuations(self.model, prompt(), missing)
        extra = dict(self.candidates, unknown=[12])
        with self.assertRaises(CandidateScoringError):
            score_candidate_continuations(self.model, prompt(), extra)

    def test_rejects_empty_or_invalid_candidate_token_ids(self):
        cases = ([], [True], [-1], [16], ["7"])
        for invalid in cases:
            with self.subTest(invalid=invalid):
                candidates = dict(self.candidates)
                candidates["glass"] = invalid
                with self.assertRaises(CandidateScoringError):
                    score_candidate_continuations(self.model, prompt(), candidates)

    def test_rejects_empty_wrong_batch_padded_or_misaligned_prompt(self):
        invalid_inputs = (
            {"input_ids": torch.empty((1, 0), dtype=torch.long)},
            {"input_ids": torch.tensor([[1, 2], [3, 4]], dtype=torch.long)},
            {"input_ids": torch.tensor([[1, 2]], dtype=torch.long),
             "attention_mask": torch.tensor([[1, 0]], dtype=torch.long)},
            {"input_ids": torch.tensor([[1, 2]], dtype=torch.long),
             "attention_mask": torch.ones((1, 3), dtype=torch.long)},
            {"input_ids": torch.tensor([[1, 2]], dtype=torch.long),
             "position_ids": torch.tensor([[0, 1]], dtype=torch.long)},
        )
        for inputs in invalid_inputs:
            with self.subTest(keys=tuple(inputs)):
                with self.assertRaises(CandidateScoringError):
                    score_candidate_continuations(self.model, inputs, self.candidates)

    def test_rejects_wrong_logit_alignment_and_nonfinite_model_output(self):
        with self.assertRaises(CandidateScoringError):
            score_candidate_continuations(FixedLogitModel(trim_logits=True).eval(), prompt(), self.candidates)
        with self.assertRaises(CandidateScoringError):
            score_candidate_continuations(FixedLogitModel(nonfinite=True).eval(), prompt(), self.candidates)

    def test_rejects_training_mode(self):
        model = FixedLogitModel().train()
        with self.assertRaises(CandidateScoringError):
            score_candidate_continuations(model, prompt(), self.candidates)


if __name__ == "__main__":
    unittest.main()
