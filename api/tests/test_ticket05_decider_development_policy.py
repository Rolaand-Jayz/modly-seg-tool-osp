import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "runtime/adapters/parts/decider_development_policy.py"
SPEC = importlib.util.spec_from_file_location("ticket05_decider_development_policy", MODULE_PATH)
assert SPEC and SPEC.loader
policy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(policy)
calibrate, decide, REJECT_ALL = policy.calibrate, policy.decide, policy.REJECT_ALL


def test_decider_precedence_and_ties():
    p = {k: .03571428571428571 for k in "ABCDEFGHIJ"}
    p["A"], p["I"], p["J"] = .25, .25, .25
    assert decide(p, .2, .2)["selected_option"] == "I"
    assert decide(p, .3, .3)["selected_option"] == "A"


def test_threshold_search_uses_observed_values_and_reject_all():
    raw = [
        {"object_id": "o1", "part_id": "p1", "probabilities": dict(A=.58,B=.06,C=.06,D=.06,E=.06,F=.06,G=.06,H=.06,I=.0,J=.0)},
        {"object_id": "o2", "part_id": "p2", "probabilities": dict(A=.06,B=.58,C=.06,D=.06,E=.06,F=.06,G=.06,H=.06,I=.0,J=.0)},
        {"object_id": "o3", "part_id": "p3", "probabilities": dict(A=.0525,B=.0525,C=.0525,D=.0525,E=.0525,F=.0525,G=.0525,H=.0525,I=.58,J=.0)},
        {"object_id": "o4", "part_id": "p4", "probabilities": dict(A=.0525,B=.0525,C=.0525,D=.0525,E=.0525,F=.0525,G=.0525,H=.0525,I=.0,J=.58)},
    ]
    truth = [
        {"object_id":"o1","part_id":"p1","truth_state":"supported","label":"A"},
        {"object_id":"o2","part_id":"p2","truth_state":"supported","label":"B"},
        {"object_id":"o3","part_id":"p3","truth_state":"unknown","label":None},
        {"object_id":"o4","part_id":"p4","truth_state":"ambiguous","label":None},
    ]
    result = calibrate(raw, truth, unknown_floor=1, ambiguous_floor=1)
    assert result["feasible"]
    assert result["threshold_i"] == .58 and result["threshold_j"] == .58
    assert result["metrics"]["selective_accuracy"] == 1
    assert REJECT_ALL in result["threshold_candidates"]["I"]


def test_no_ood_feasible_policy_rejects():
    raw = [{"object_id":"o","part_id":"p","probabilities":dict(A=.07,B=.07,C=.07,D=.07,E=.07,F=.07,G=.07,H=.07,I=.08,J=.36)}]
    truth = [{"object_id":"o","part_id":"p","truth_state":"supported","label":"A"}]
    result = calibrate(raw, truth)
    assert result["feasible"] is False
