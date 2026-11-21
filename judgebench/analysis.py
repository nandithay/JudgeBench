"""Join only matching IDs; export descriptive findings with denominators and missingness."""
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import agreement, position_metrics, reviewer_metrics
from .pipeline import manifest_for
from .review import human_records
from .rubrics import weighted_score
from .schemas import DIMENSIONS, HumanRating, JudgeRating, Scores
from .storage import now, records, write_json


def joined_ratings(run):
    manifest = manifest_for(run)
    questions = {q["id"]: q for q in manifest["questions"]}
    responses = {r["response_id"]: r for r in records(Path(run) / "responses")}
    judgments = {r["response_id"]: r for r in records(Path(run) / "judgments" / "baseline")}
    rows = []
    for human in human_records(run):
        # Audit metadata is not part of the annotation input schema.
        HumanRating.model_validate({k: v for k, v in human.items() if k not in ("created_at", "experiment_id")})
        rid = human["response_id"]
        if human["experiment_id"] != manifest["experiment_id"]:
            raise ValueError("Annotation belongs to a different experiment.")
        if rid not in responses:
            raise ValueError("Annotation references an unknown response.")
        if rid not in judgments:
            continue
        judge = judgments[rid]
        JudgeRating.model_validate({k: judge[k] for k in ("scores", "reasons", "p_good")})
        response = responses[rid]
        q = questions[response["question_id"]]
        row = {"response_id": rid, "question_id": q["id"], "category": q["category"],
               "split": q["split"], "generator_index": response["generator_index"],
               "reviewer": human["reviewer"], "p_good": judge["p_good"],
               "human_good": Scores(**human["scores"]).good(), "judge_good": Scores(**judge["scores"]).good(),
               "tags": human["tags"], "notes": human["notes"], "response": response["text"],
               "prompt": q["prompt"], "reference": q["reference"], "judge_reasons": judge["reasons"]}
        for dimension in DIMENSIONS:
            row[f"human_{dimension}"] = human["scores"][dimension]
            row[f"judge_{dimension}"] = judge["scores"][dimension]
            row[f"gap_{dimension}"] = judge["scores"][dimension] - human["scores"][dimension]
        row["max_abs_gap"] = max(abs(row[f"gap_{d}"]) for d in DIMENSIONS)
        row["quality_disagreement"] = row["human_good"] != row["judge_good"]
        rows.append(row)
    return pd.DataFrame(rows)


def compare_humans(frame):
    output = []
    if frame.empty:
        return output
    for a, b in itertools.combinations(sorted(frame.reviewer.unique()), 2):
        left = frame[frame.reviewer == a]
        right = frame[frame.reviewer == b]
        common = left.merge(right, on="response_id", suffixes=("_a", "_b"), validate="one_to_one")
        for dimension in DIMENSIONS:
            output.append({"reviewer_a": a, "reviewer_b": b, "dimension": dimension,
                           **agreement(common[f"human_{dimension}_a"], common[f"human_{dimension}_b"])})
    return output


def human_comparison_frame(run):
    """Human–human reliability must not depend on the availability of LLM scores."""
    manifest = manifest_for(run)
    rows = []
    for human in human_records(run):
        if human["experiment_id"] != manifest["experiment_id"]:
            raise ValueError("Human annotation belongs to a different run.")
        rows.append({"response_id": human["response_id"], "reviewer": human["reviewer"],
                     **{f"human_{d}": human["scores"][d] for d in DIMENSIONS}})
    return pd.DataFrame(rows)


def sensitivity(run):
    run = Path(run)
    responses = {r["response_id"]: r for r in records(run / "responses")}
    baseline = {r["response_id"]: r for r in records(run / "judgments" / "baseline")}
    strict = {r["response_id"]: r for r in records(run / "judgments" / "strict_correctness")}
    equal = dict.fromkeys(DIMENSIONS, 0.25)
    weights = {"correctness": 0.55, "relevance": 0.1, "completeness": 0.1, "instruction_following": 0.25}
    deltas = []
    for rid in sorted(baseline.keys() & strict.keys()):
        base, variant = baseline[rid]["scores"], strict[rid]["scores"]
        row = {"response_id": rid, "question_id": responses[rid]["question_id"],
               "baseline": weighted_score(base, equal), "strict": weighted_score(variant, equal)}
        row.update({f"delta_{d}": variant[d] - base[d] for d in DIMENSIONS})
        row["delta_mean"] = row["strict"] - row["baseline"]
        deltas.append(row)
    # Per-question rankings compare the SAME two candidate responses under each condition.
    pair_rows = []
    for qid in sorted({r["question_id"] for r in responses.values()}):
        pair = sorted([r for r in responses.values() if r["question_id"] == qid], key=lambda r: r["generator_index"])
        if len(pair) != 2 or any(r["response_id"] not in baseline for r in pair):
            continue
        ids = [r["response_id"] for r in pair]

        def winner(table, w):
            gap = weighted_score(table[ids[0]]["scores"], w) - weighted_score(table[ids[1]]["scores"], w)
            return "tie" if abs(gap) < 1e-9 else str(pair[0 if gap > 0 else 1]["generator_index"])

        original = winner(baseline, equal)
        for condition, table, w in [("weights_only", baseline, weights), ("criteria_only", strict, equal)]:
            if any(rid not in table for rid in ids):
                continue
            changed = winner(table, w)
            pair_rows.append({"question_id": qid, "condition": condition, "baseline_winner": original,
                              "variant_winner": changed, "ranking_changed": original != changed,
                              "outright_reversal": original != changed and "tie" not in (original, changed)})
    summary = {"matched_responses": len(deltas), "weight_variant": weights, "conditions": {}}
    if deltas:
        summary["mean_score_delta"] = float(np.mean([r["delta_mean"] for r in deltas]))
        summary["mean_dimension_deltas"] = {d: float(np.mean([r[f"delta_{d}"] for r in deltas])) for d in DIMENSIONS}
    for condition in ("weights_only", "criteria_only"):
        group = [r for r in pair_rows if r["condition"] == condition]
        summary["conditions"][condition] = {"paired_questions": len(group),
            "ranking_change_rate": float(np.mean([r["ranking_changed"] for r in group])) if group else None,
            "outright_reversal_rate": float(np.mean([r["outright_reversal"] for r in group])) if group else None}
    return summary, deltas, pair_rows


def export_table(folder, name, rows):
    folder = Path(folder)
    if isinstance(rows, pd.DataFrame):
        rows = rows.to_dict("records")
    write_json(folder / f"{name}.json", rows)
    frame = pd.DataFrame(rows)
    for column in frame.columns:
        frame[column] = frame[column].map(lambda v: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v)
    folder.mkdir(parents=True, exist_ok=True)
    frame.to_csv(folder / f"{name}.csv", index=False)


def analyze(run):
    run = Path(run)
    manifest = manifest_for(run)
    output = run / "exports"
    responses = records(run / "responses")
    human = human_records(run)
    judgments = records(run / "judgments" / "baseline")
    frame = joined_ratings(run)
    summary = {"generated_at": now(), "experiment_id": manifest["experiment_id"],
               "coverage": {"questions": 40, "expected_responses": 80, "responses": len(responses),
                            "baseline_judgments": len(judgments), "human_annotations": len(human),
                            "matched_annotations": len(frame)}, "reviewers": {}, "category_metrics": [],
               "limitations": ["Small curated dataset, not representative of general model quality.",
                               "Human ratings are a fallible reference, not objective truth.",
                               "Bootstrap intervals resample questions, not correlated response rows.",
                               "All results describe the saved models, prompts, and available annotations only.",
                               "Same-order variability is a control, not a full causal correction.",
                               "Rubric changes may alter the quality construct, not just evaluator reliability."]}
    predictions = []
    disagreements = frame[(frame.max_abs_gap >= 2) | frame.quality_disagreement].copy() if len(frame) else frame
    if len(frame):
        for reviewer, group in frame.groupby("reviewer"):
            metrics, pred = reviewer_metrics(group, manifest["config"]["seed"])
            # Headline agreement is held-out only; dev is explicitly exploratory.
            heldout, _ = reviewer_metrics(group[group.split == "test"], manifest["config"]["seed"])
            metrics["heldout_test"] = heldout
            summary["reviewers"][reviewer] = metrics
            predictions.extend({"reviewer": reviewer, **r} for r in pred)
            for (split, category), subset in group.groupby(["split", "category"]):
                for d in DIMENSIONS:
                    summary["category_metrics"].append({"reviewer": reviewer, "split": split,
                        "category": category, "dimension": d,
                        **agreement(subset[f"human_{d}"], subset[f"judge_{d}"])})
    tag_rows = []
    if len(disagreements):
        for (reviewer, split), group in disagreements.groupby(["reviewer", "split"]):
            counts = group.explode("tags").tags.dropna().value_counts()
            tag_rows.extend({"reviewer": reviewer, "split": split, "tag": tag, "count": int(n)} for tag, n in counts.items())
    position_rows = records(run / "position")
    summary["position"], pairs, controls = position_metrics(position_rows)
    # Separate held-out report to avoid silently pooling dev and test experiments.
    test_ids = {q["id"] for q in manifest["questions"] if q["split"] == "test"}
    summary["position"]["heldout_test"], _, _ = position_metrics([r for r in position_rows if r["question_id"] in test_ids])
    summary["sensitivity"], deltas, rankings = sensitivity(run)
    summary["human_human"] = compare_humans(human_comparison_frame(run))
    tables = {"responses": responses, "human_ratings": human, "judge_baseline": judgments,
              "judge_strict": records(run / "judgments" / "strict_correctness"),
              "matched_ratings": frame, "disagreements": disagreements,
              "disagreement_tags": tag_rows, "calibration_predictions": predictions,
              "position_judgments": position_rows, "position_pairs": pairs, "same_order_controls": controls,
              "rubric_score_deltas": deltas, "rubric_rankings": rankings,
              "human_human": summary["human_human"], "category_metrics": summary["category_metrics"]}
    for name, rows in tables.items():
        export_table(output, name, rows)
    write_json(output / "metrics.json", summary)
    write_json(output / "manifest.json", manifest)
    # Latency/token records come from actual provider payloads, not estimates.
    requests = []
    for cached in records(run / "cache"):
        raw = cached.get("raw", {})
        requests.append({"request_id": cached["request_id"], "trial": cached["request"]["trial"],
                         "requested_model": cached["request"]["model"]["model"],
                         "returned_model": raw.get("model", raw.get("model_version")),
                         "latency_seconds": cached.get("latency_seconds"),
                         "attempts": cached.get("attempts"),
                         "usage": raw.get("usage", raw.get("usage_metadata")),
                         "complete": cached.get("complete")})
    export_table(output, "api_usage", requests)
    lines = ["# JudgeBench findings", "", f"Experiment: `{manifest['experiment_id']}`", "",
             f"Saved coverage: {len(responses)}/80 responses; {len(judgments)}/80 baseline judgments; "
             f"{len(human)} human annotations; {len(frame)} matched annotations.", ""]
    if not len(frame):
        lines += ["No matched human/judge ratings exist. Human agreement, binary quality performance, "
                  "and calibration findings cannot yet be reported.", ""]
    for reviewer, metrics in summary["reviewers"].items():
        test = metrics["heldout_test"]
        lines += [f"## Reviewer {reviewer}", "", f"Matched responses: {metrics['n']} overall; {test['n']} held-out test.", ""]
        if test["n"]:
            for d, values in test["dimensions"].items():
                lines.append(f"- {d}: exact agreement {values['exact_agreement']:.3f}; "
                             f"kappa {values['kappa']}; quadratic weighted kappa {values['weighted_kappa']}; n={values['n']}.")
            lines += ["", f"Held-out binary quality metrics: {json.dumps(test['quality'])}", ""]
        lines += [f"Calibration: {json.dumps(metrics['calibration'])}", ""]
    lines += ["## Position bias", "", json.dumps(summary["position"], indent=2), "",
              "Swapped-order changes must be interpreted alongside same-order variability; ties are explicit.", "",
              "## Rubric sensitivity", "", json.dumps(summary["sensitivity"], indent=2), "",
              "## Disagreements", "", f"{len(disagreements)} matched annotations have an absolute score gap >=2 "
              "or a binary quality mismatch. See disagreements.csv. Tags are reviewer-assigned response issues, "
              "not automatically proven judge error causes.", "", "## Limitations", ""]
    lines.extend(f"- {item}" for item in summary["limitations"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
