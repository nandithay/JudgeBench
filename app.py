"""Run with: streamlit run app.py. This UI makes zero LLM API calls."""
import hashlib
import io
import random
import re
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from judgebench.analysis import analyze
from judgebench.pipeline import manifest_for
from judgebench.review import human_records, save_review
from judgebench.schemas import DIMENSIONS, ERROR_TAGS
from judgebench.storage import read_json, records

st.set_page_config(page_title="JudgeBench", page_icon="⚖️", layout="wide")
st.title("JudgeBench")
st.caption("LLM judge reliability · calibration · position bias")
run = Path(st.sidebar.text_input("Run folder", "runs/first"))
mode = st.sidebar.radio("Page", ["Overview", "Human review", "Agreement & calibration", "Position bias", "Rubric sensitivity", "Disagreements", "Exports"])
st.sidebar.caption("Local, single-machine app. No API calls are made by this dashboard.")
if not (run / "manifest.json").exists():
    st.info("Initialize a run first: python -m judgebench.cli init --run runs/first")
    st.stop()
try:
    manifest = manifest_for(run)
    responses = records(run / "responses")
    humans = human_records(run)
except (ValueError, OSError) as exc:
    st.error(str(exc))
    st.stop()

questions = {q["id"]: q for q in manifest["questions"]}


def show_rubric():
    rubric = manifest["rubrics"]["baseline"]
    st.write(rubric["guidance"])
    st.dataframe(pd.DataFrame(rubric["anchors"]).rename_axis("Score"), use_container_width=True)


def load_metrics():
    path = run / "exports" / "metrics.json"
    if not path.exists():
        st.info("Generate analysis exports from the sidebar or CLI after collecting ratings.")
        st.stop()
    metrics = read_json(path)
    st.caption(f"Analysis snapshot: {metrics['generated_at']}. Recompute after adding ratings.")
    return metrics


def figure_done(fig):
    st.pyplot(fig)
    plt.close(fig)


if mode != "Human review" and st.sidebar.button("Recompute analysis (no API calls)"):
    with st.spinner("Computing metrics and exports…"):
        analyze(run)
    st.sidebar.success("Analysis saved.")

if mode == "Overview":
    st.subheader("Run coverage")
    a, b, c, d = st.columns(4)
    a.metric("Questions", len(questions))
    b.metric("Responses", f"{len(responses)} / 80")
    c.metric("Baseline judgments", f"{len(records(run / 'judgments' / 'baseline'))} / 80")
    d.metric("Human annotations", len(humans))
    st.write("Review responses independently before viewing judge scores. Use a second reviewer to measure human–human agreement.")
    st.dataframe(pd.DataFrame(manifest["questions"])[["id", "category", "split", "prompt"]], hide_index=True)
    with st.expander("Baseline rubric"):
        show_rubric()
    with st.expander("Model settings and reproducibility"):
        st.json(manifest["config"])
        st.code(manifest["experiment_id"])
    st.warning("No scores or study findings are prefilled. Dataset references and fictional passages are not model evaluation results.")

elif mode == "Human review":
    st.subheader("Independent human review")
    reviewer = st.text_input("Reviewer ID (letters, numbers, _ or -)", placeholder="reviewer_1")
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", reviewer):
        st.info("Enter a reviewer ID to start. Use the same ID to resume your reviews.")
        st.stop()
    # Deterministic per-reviewer shuffle; never reveal model names, judge scores, or other annotations.
    seed = int(hashlib.sha256(f"{manifest['config']['seed']}:{reviewer}".encode()).hexdigest()[:8], 16)
    queue = sorted(responses, key=lambda r: r["response_id"])
    random.Random(seed).shuffle(queue)
    reviewed = {h["response_id"]: h for h in humans if h["reviewer"] == reviewer}
    st.progress(len(reviewed) / max(len(queue), 1), text=f"{len(reviewed)} / {len(queue)} available responses reviewed")
    revising = st.checkbox("Revise a saved annotation")
    available = [r for r in queue if (r["response_id"] in reviewed) == revising]
    if not available:
        st.success("No responses in this queue. Generate more responses or switch queue.")
        st.stop()
    # Only anonymous queue numbers appear in selection labels.
    selected = st.selectbox("Response", range(len(available)), format_func=lambda i: f"Response {i+1}")
    response = available[selected]
    q = questions[response["question_id"]]
    st.write("**Task**")
    st.text(q["prompt"])
    st.write("**Candidate response**")
    st.text(response["text"])
    with st.expander("Reference and checklist", expanded=True):
        st.text(q["reference"])
        for item in q["checklist"]:
            st.write(f"• {item}")
        st.caption("Accept correct alternatives. Flag reference issues rather than forcing an exact match.")
    with st.expander("Scoring rubric", expanded=False):
        show_rubric()
    old = reviewed.get(response["response_id"], {})
    with st.form(f"review-{reviewer}-{response['response_id']}"):
        values = {d: st.selectbox(d.replace("_", " ").title(), [None, 1, 2, 3, 4, 5],
                                 index=old.get("scores", {}).get(d, 0),
                                 format_func=lambda x: "Select a score" if x is None else str(x)) for d in DIMENSIONS}
        tags = st.multiselect("Observed response issues (optional)", ERROR_TAGS, default=old.get("tags", []))
        notes = st.text_area("Evidence / notes", value=old.get("notes", ""))
        submitted = st.form_submit_button("Save annotation")
    if submitted:
        if any(v is None for v in values.values()):
            st.error("Select all four scores before saving.")
        else:
            try:
                save_review(run, {"response_id": response["response_id"], "reviewer": reviewer,
                                  "scores": values, "notes": notes, "tags": tags})
                st.rerun()
            except (ValueError, OSError) as exc:
                st.error(str(exc))

elif mode == "Agreement & calibration":
    metrics = load_metrics()
    if not metrics["reviewers"]:
        st.info("No matched human and judge ratings yet.")
        st.stop()
    reviewer = st.selectbox("Reviewer", sorted(metrics["reviewers"]))
    overall = metrics["reviewers"][reviewer]
    split = st.radio("Agreement subset", ["Held-out test", "All annotated (exploratory)"])
    values = overall["heldout_test"] if split == "Held-out test" else overall
    st.metric("Matched responses", values["n"])
    if values["n"]:
        table = pd.DataFrame(values["dimensions"]).T
        st.dataframe(table.drop(columns="exact_ci95"), use_container_width=True)
        st.caption("Kappa adjusts for chance; weighted kappa uses quadratic penalties. Undefined kappa is blank, not zero. Scores are ordinal.")
        with st.expander("Question-cluster bootstrap intervals"):
            st.json({d: v["exact_ci95"] for d, v in values["dimensions"].items()})
        st.write("**Binary quality: all four scores ≥4**")
        st.json(values["quality"])
        matrix = values["quality"]["confusion_matrix"]
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.imshow(matrix, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(matrix[i][j]), ha="center", va="center")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Not good", "Good"], yticklabels=["Not good", "Good"], xlabel="Judge", ylabel="Human")
        figure_done(fig)
    st.subheader("Calibration on held-out responses")
    cal = overall["calibration"]
    st.write(cal["status"])
    st.json({k: v for k, v in cal.items() if k not in ("bins", "calibrated_bins")})
    if cal["bins"]:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
        for key, label in [("bins", "Raw judge"), ("calibrated_bins", "Dev-fitted isotonic")]:
            bins = cal[key]
            if bins:
                ax.plot([b["mean_probability"] for b in bins], [b["human_good_rate"] for b in bins], "o-", label=label)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Predicted probability of human-good", ylabel="Observed human-good fraction")
        ax.legend()
        figure_done(fig)
        st.dataframe(pd.DataFrame(cal["bins"]))
    st.subheader("Human–human agreement (matched annotated responses)")
    st.dataframe(pd.DataFrame(metrics["human_human"]))
    st.subheader("Category breakdown")
    categories = [r for r in metrics["category_metrics"] if r["reviewer"] == reviewer]
    st.dataframe(pd.DataFrame(categories))

elif mode == "Position bias":
    metrics = load_metrics()
    st.subheader("Order-swap experiment")
    scope = st.radio("Subset", ["Held-out test", "All questions (exploratory)"])
    values = metrics["position"]["heldout_test"] if scope == "Held-out test" else {k: v for k, v in metrics["position"].items() if k != "heldout_test"}
    st.json(values)
    if values.get("paired_trials"):
        rates = {"Swap flip": values["flip_rate"], "Outright reversal": values["outright_reversal_rate"],
                 "Tie transition": values["tie_transition_rate"], "Same-order change": values.get("same_order_change_rate")}
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.barh(list(rates), [0 if v is None else v for v in rates.values()])
        ax.set(xlim=(0, 1), xlabel="Rate (see denominators above)")
        figure_done(fig)
    st.caption("First position means displayed A. Winners are mapped back to response identity before measuring flips. An all-tie judge is consistent but uninformative.")
    trials = records(run / "position")
    st.dataframe(pd.DataFrame(trials))
    if trials:
        qid = st.selectbox("Inspect a question", sorted({r["question_id"] for r in trials}))
        st.text(questions[qid]["prompt"])
        for r in sorted([r for r in responses if r["question_id"] == qid], key=lambda r: r["generator_index"]):
            st.write(f"Generator {r['generator_index']}")
            st.text(r["text"])
        st.dataframe(pd.DataFrame([r for r in trials if r["question_id"] == qid]))

elif mode == "Rubric sensitivity":
    metrics = load_metrics()
    st.subheader("Criteria changes and weight changes")
    st.write("Criteria-only: rerun the same responses with unsupported claims / invalid reasoning capping correctness at 2. Weights-only: reuse baseline scores and increase correctness/instruction weights. Ranking changes include ties and are reported separately from outright reversals.")
    st.json(metrics["sensitivity"])
    path = run / "exports" / "rubric_score_deltas.json"
    deltas = read_json(path) if path.exists() else []
    if deltas:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist([r["delta_mean"] for r in deltas], bins=9)
        ax.set(xlabel="Strict − baseline mean score", ylabel="Responses")
        figure_done(fig)
        st.dataframe(pd.DataFrame(deltas))
    rankings = run / "exports" / "rubric_rankings.json"
    if rankings.exists():
        st.dataframe(pd.DataFrame(read_json(rankings)))

elif mode == "Disagreements":
    load_metrics()
    path = run / "exports" / "disagreements.json"
    rows = read_json(path)
    st.write("Score gap ≥2 on any dimension, or different binary quality labels. Review evidence before assigning an evaluator error.")
    if not rows:
        st.info("No disagreements available in the saved analysis.")
        st.stop()
    reviewer = st.selectbox("Reviewer", sorted({r["reviewer"] for r in rows}))
    category = st.selectbox("Category", ["All"] + sorted({r["category"] for r in rows}))
    filtered = [r for r in rows if r["reviewer"] == reviewer and (category == "All" or r["category"] == category)]
    filtered.sort(key=lambda r: r["max_abs_gap"], reverse=True)
    st.dataframe(pd.DataFrame(filtered)[["question_id", "split", "max_abs_gap", "quality_disagreement", "tags", "notes"]])
    if filtered:
        index = st.selectbox("Inspect disagreement", range(len(filtered)))
        row = filtered[index]
        st.text(row["prompt"])
        st.text(row["response"])
        st.write("Reference:")
        st.text(row["reference"])
        st.dataframe(pd.DataFrame({"human": {d: row[f"human_{d}"] for d in DIMENSIONS},
                                   "judge": {d: row[f"judge_{d}"] for d in DIMENSIONS}}))
        st.json(row["judge_reasons"])
        st.caption("Issue tags were entered independently by the reviewer; they are candidate patterns, not proven causes.")

elif mode == "Exports":
    load_metrics()
    st.write("Exports contain actual saved records only. Raw provider outputs and full prompts remain in the local run cache.")
    files = sorted((run / "exports").glob("*"))
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            if path.is_file():
                archive.writestr(path.name, path.read_bytes())
    st.download_button("Download analysis ZIP", buffer.getvalue(), file_name="judgebench-results.zip", mime="application/zip")
    for path in files:
        if path.is_file():
            st.download_button(path.name, path.read_bytes(), file_name=path.name, key=path.name)
