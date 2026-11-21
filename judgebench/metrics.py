"""Pure statistical functions: human labels are the reference, never judge labels."""
import warnings

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (brier_score_loss, cohen_kappa_score, confusion_matrix,
                             precision_recall_fscore_support)

from .schemas import DIMENSIONS


def finite(value):
    return float(value) if np.isfinite(value) else None


def agreement(human, judge):
    h, j = np.asarray(human), np.asarray(judge)
    if len(h) == 0:
        return {"n": 0, "exact_agreement": None, "kappa": None, "weighted_kappa": None,
                "mae": None, "judge_minus_human": None}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        k = cohen_kappa_score(h, j, labels=[1, 2, 3, 4, 5])
        wk = cohen_kappa_score(h, j, labels=[1, 2, 3, 4, 5], weights="quadratic")
    return {"n": len(h), "exact_agreement": float(np.mean(h == j)), "kappa": finite(k),
            "weighted_kappa": finite(wk), "mae": float(np.mean(np.abs(h - j))),
            "judge_minus_human": float(np.mean(j - h))}


def binary_metrics(human_good, judge_good):
    h, j = np.asarray(human_good, dtype=int), np.asarray(judge_good, dtype=int)
    if not len(h):
        return {"n": 0}
    precision, recall, f1, _ = precision_recall_fscore_support(h, j, average="binary", zero_division=0)
    return {"n": len(h), "human_positive": int(h.sum()), "judge_positive": int(j.sum()),
            "precision": float(precision), "recall": float(recall), "f1": float(f1),
            "exact_agreement": float(np.mean(h == j)),
            "confusion_matrix": confusion_matrix(h, j, labels=[0, 1]).tolist(),
            "undefined_precision": bool(j.sum() == 0), "undefined_recall": bool(h.sum() == 0),
            "note": "Undefined precision/recall use 0 by convention; flags identify these cases."}


def cluster_ci(frame, statistic, seed=2026, iterations=500):
    """Resample questions, retaining both responses from each sampled question."""
    ids = frame.question_id.unique() if len(frame) else []
    if len(ids) < 2:
        return None
    rng = np.random.default_rng(seed)
    groups = {qid: group for qid, group in frame.groupby("question_id")}
    values = []
    for _ in range(iterations):
        sample = pd.concat([groups[qid] for qid in rng.choice(ids, len(ids), replace=True)], ignore_index=True)
        value = statistic(sample)
        if value is not None and np.isfinite(value):
            values.append(value)
    if not values:
        return None
    return {"low": float(np.quantile(values, 0.025)), "high": float(np.quantile(values, 0.975)),
            "valid_resamples": len(values), "unit": "question", "iterations": iterations}


def reliability_bins(y, probabilities, bins=5):
    y, p = np.asarray(y, dtype=int), np.asarray(probabilities, dtype=float)
    rows = []
    indices = np.minimum((p * bins).astype(int), bins - 1)
    for index in range(bins):
        mask = indices == index
        if mask.any():
            rows.append({"bin": index, "n": int(mask.sum()), "mean_probability": float(p[mask].mean()),
                         "human_good_rate": float(y[mask].mean())})
    return rows


def calibration(frame):
    """Fit isotonic only on dev labels. Never tune on test annotations."""
    dev = frame[frame.split == "dev"]
    test = frame[frame.split == "test"].copy()
    result = {"dev_n": len(dev), "test_n": len(test), "raw_brier": None,
              "calibrated_brier": None, "bins": [], "calibrated_bins": [],
              "status": "No held-out human labels available."}
    predictions = []
    if test.empty:
        return result, predictions
    y = test.human_good.astype(int).to_numpy()
    raw = test.p_good.to_numpy(dtype=float)
    result.update(raw_brier=float(brier_score_loss(y, raw)), bins=reliability_bins(y, raw),
                  status="Raw held-out calibration only; need >=10 dev responses and both dev classes to fit isotonic.")
    result["ece_5_bins"] = float(sum(r["n"] * abs(r["mean_probability"] - r["human_good_rate"])
                                       for r in result["bins"]) / len(test))
    adjusted = None
    if len(dev) >= 10 and dev.human_good.nunique() == 2 and dev.p_good.nunique() >= 2:
        model = IsotonicRegression(out_of_bounds="clip").fit(dev.p_good, dev.human_good.astype(int))
        adjusted = model.predict(raw)
        result.update(calibrated_brier=float(brier_score_loss(y, adjusted)),
                      calibrated_bins=reliability_bins(y, adjusted),
                      status="Isotonic fitted on dev only; both Brier scores measured on test.")
    for index, (_, row) in enumerate(test.iterrows()):
        predictions.append({"response_id": row.response_id, "human_good": int(y[index]),
                            "raw_probability": float(raw[index]),
                            "calibrated_probability": float(adjusted[index]) if adjusted is not None else None})
    return result, predictions


def reviewer_metrics(frame, seed=2026):
    if frame.empty:
        return {"n": 0}, []
    dimensions = {}
    for d in DIMENSIONS:
        dimensions[d] = agreement(frame[f"human_{d}"], frame[f"judge_{d}"])
        dimensions[d]["exact_ci95"] = cluster_ci(
            frame, lambda f: float((f[f"human_{d}"] == f[f"judge_{d}"]).mean()), seed)
    quality = binary_metrics(frame.human_good, frame.judge_good)
    quality["agreement_ci95"] = cluster_ci(frame, lambda f: float((f.human_good == f.judge_good).mean()), seed)
    cal, predictions = calibration(frame)
    return {"n": len(frame), "questions": int(frame.question_id.nunique()),
            "dimensions": dimensions, "quality": quality, "calibration": cal}, predictions


def position_metrics(rows):
    """Do not interpret incomplete trials as consistent, inconsistent, or ties."""
    if not rows:
        return {"judgments": 0, "paired_trials": 0}, [], []
    frame = pd.DataFrame(rows)
    pairs = []
    for (qid, repeat), group in frame.groupby(["question_id", "repeat"]):
        orders = {r["order"]: r for r in group.to_dict("records")}
        if set(orders) != {"AB", "BA"}:
            continue
        a, b = orders["AB"]["winner_response_id"], orders["BA"]["winner_response_id"]
        pairs.append({"question_id": qid, "repeat": int(repeat), "AB_winner": a, "BA_winner": b,
                      "consistent": a == b, "reversal": a != b and "tie" not in (a, b),
                      "tie_transition": a != b and "tie" in (a, b)})
    # Compare adjacent repetitions within the same order, as a stochastic baseline.
    controls = []
    for (qid, order), group in frame.groupby(["question_id", "order"]):
        group = group.sort_values("repeat").to_dict("records")
        for previous, current in zip(group, group[1:]):
            if current["repeat"] != previous["repeat"] + 1:
                continue
            controls.append({"question_id": qid, "order": order, "repeat": current["repeat"],
                             "changed": previous["winner_response_id"] != current["winner_response_id"]})
    decisive = frame[frame.selected_position != "tie"]
    summary = {"judgments": len(frame), "paired_trials": len(pairs),
               "paired_questions": len({p["question_id"] for p in pairs}),
               "tie_rate": float((frame.selected_position == "tie").mean()),
               "first_position_selection_rate_all": float((frame.selected_position == "A").mean()),
               "first_position_selection_rate_decisive": float((decisive.selected_position == "A").mean()) if len(decisive) else None,
               "decisive_n": len(decisive), "same_order_comparisons": len(controls),
               "same_order_change_rate": float(np.mean([c["changed"] for c in controls])) if controls else None,
               "note": "Swapped-order changes alone do not establish causal position bias; inspect same-order variability."}
    for field, name in [("consistent", "preference_consistency"), ("reversal", "outright_reversal_rate"),
                        ("tie_transition", "tie_transition_rate")]:
        summary[name] = float(np.mean([p[field] for p in pairs])) if pairs else None
    summary["flip_rate"] = 1 - summary["preference_consistency"] if pairs else None
    if pairs:
        summary["flip_ci95"] = cluster_ci(pd.DataFrame(pairs), lambda f: float((~f.consistent).mean()))
    return summary, pairs, controls
