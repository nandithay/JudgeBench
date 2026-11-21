import pandas as pd
import pytest

from judgebench.metrics import (agreement, binary_metrics, calibration, cluster_ci,
                               position_metrics)


def test_agreement_known_answers():
    result = agreement([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    assert result["exact_agreement"] == result["kappa"] == result["weighted_kappa"] == 1
    assert result["mae"] == 0


def test_constant_kappa_is_undefined_not_perfect():
    result = agreement([5, 5], [5, 5])
    assert result["exact_agreement"] == 1
    assert result["kappa"] is None
    assert agreement([], [])["n"] == 0


def test_binary_reference_direction():
    result = binary_metrics([1, 1, 1, 0], [1, 0, 0, 0])
    assert result["precision"] == 1
    assert result["recall"] == pytest.approx(1/3)
    assert result["f1"] == 0.5
    assert result["confusion_matrix"] == [[1, 0], [2, 1]]
    assert binary_metrics([0, 0], [0, 0])["undefined_recall"]


def row(order, repeat, selected, winner, qid="q"):
    return dict(question_id=qid, order=order, repeat=repeat,
                selected_position=selected, winner_response_id=winner)


def test_order_mapping_consistency_is_about_identity():
    summary, pairs, _ = position_metrics([row("AB", 0, "A", "response0"), row("BA", 0, "B", "response0")])
    assert summary["preference_consistency"] == 1
    assert summary["first_position_selection_rate_decisive"] == 0.5


def test_always_first_judge_reverses_identity():
    rows = [row("AB", k, "A", "response0") for k in range(2)] + [row("BA", k, "A", "response1") for k in range(2)]
    result, _, _ = position_metrics(rows)
    assert result["flip_rate"] == result["outright_reversal_rate"] == 1
    assert result["same_order_change_rate"] == 0


def test_ties_and_incomplete_trials():
    result, pairs, _ = position_metrics([row("AB", 0, "tie", "tie"), row("BA", 0, "A", "response1"), row("AB", 1, "A", "response0")])
    assert result["paired_trials"] == 1
    assert result["tie_transition_rate"] == 1
    assert result["outright_reversal_rate"] == 0
    all_ties, _, _ = position_metrics([row("AB", 0, "tie", "tie"), row("BA", 0, "tie", "tie")])
    assert all_ties["first_position_selection_rate_decisive"] is None


def test_bootstrap_preserves_question_clusters():
    frame = pd.DataFrame({"question_id": ["a", "a", "b", "b"], "value": [0, 1, 0, 1]})
    result = cluster_ci(frame, lambda f: float(f.value.mean()), iterations=30)
    assert result["low"] == result["high"] == 0.5
    assert cluster_ci(frame.iloc[:2], lambda f: f.value.mean()) is None


def test_calibration_fits_only_dev_not_test():
    frame = pd.DataFrame([dict(response_id=f"d{i}", split="dev", p_good=i/9, human_good=i >= 5) for i in range(10)] +
                         [dict(response_id="t0", split="test", p_good=0.2, human_good=False),
                          dict(response_id="t1", split="test", p_good=0.8, human_good=True)])
    result, predictions = calibration(frame)
    assert result["raw_brier"] == pytest.approx(0.04)
    assert result["calibrated_brier"] == 0
    reversed_labels = frame.copy()
    reversed_labels.loc[reversed_labels.split == "test", "human_good"] = [True, False]
    changed, changed_predictions = calibration(reversed_labels)
    assert [r["calibrated_probability"] for r in changed_predictions] == [r["calibrated_probability"] for r in predictions]
    assert changed["calibrated_brier"] == 1


def test_calibration_handles_missing_dev_and_probability_one():
    frame = pd.DataFrame([dict(response_id="x", split="test", p_good=1.0, human_good=True)])
    result, _ = calibration(frame)
    assert result["calibrated_brier"] is None
    assert result["bins"][0]["bin"] == 4
