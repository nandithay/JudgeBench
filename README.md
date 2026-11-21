# JudgeBench

A small Python lab for asking: **When can we trust an LLM judge, and where does it disagree with people?**

Uses Python 3.10+, Pandas, Pydantic, Scikit-learn, Matplotlib, Streamlit, Groq, and Google GenAI. Data lives in JSON and CSV. No database, agent framework, or backend service.

**Study status:** no live model evaluations or human annotations are included. The 40 questions and their reference answers are authored dataset content, not experimental findings. Synthetic API outputs exist only in temporary test fixtures.

**Build validation:** seven dependency-free checks and Python compilation passed. The full test suite and live integrations could not be run in the build environment because third-party dependency downloads were unavailable. See [docs/VALIDATION.md](docs/VALIDATION.md) for the exact validation boundary.

## 1. Install

From the project folder, with Python 3.10 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m pytest -q
```

If your default Python is older, replace `python3` with `python3.10`, `python3.11`, or a newer installed interpreter. On Windows, activate with `.venv\Scripts\activate`.

The included `data/questions.json` already has 40 questions. To regenerate after deliberately removing it, use `python -m judgebench.cli dataset`. That command refuses to overwrite an existing dataset.

## 2. Configure providers

Create keys in [Groq](https://console.groq.com/keys) and [Google AI Studio](https://aistudio.google.com/api-keys). Keep keys in environment variables, never in code, dataset files, or chat.

```bash
export GROQ_API_KEY='your-local-groq-key'
export GEMINI_API_KEY='your-local-gemini-key'
```

`.env.example` documents variable names; the app does **not** automatically load `.env` files. If you create one locally, export its variables in your shell. Real `.env` files are ignored by Git and excluded from the downloadable project archive.

Edit `config.json` before initializing a run:

| Role | Initial configurable model ID | Provider |
| --- | --- | --- |
| Generator 0 | `openai/gpt-oss-20b` | Groq |
| Generator 1 | `qwen/qwen3.8-27b` | Groq |
| Judge | `gemini-3.5-flash-lite` | Gemini |

These are starting configurations, not guarantees of account access, future availability, or free quotas. Check your account's active model list and free limits. A `groq` model ID beginning `openai/` is a Groq-hosted open-weight model; it uses your **Groq** key. This project does not require an OpenAI Platform key.

All three model IDs must differ. Any role can use either implemented provider. There is no automatic model or paid fallback. A separate model family for the judge reduces one possible confound but does not prove the judge is reliable.

Use original, non-sensitive dataset content with free services. See the providers' current data-use terms, especially Google's free-tier terms. Charges depend on the account/model plan; this code cannot guarantee a free run if you enable billing.

## 3. Freeze a run and inspect the plan

```bash
python -m judgebench.cli init --run runs/first
python -m judgebench.cli plan --run runs/first
```

Initialization copies the questions, references, rubrics, prompts, model settings, and installed package versions into a content-hashed manifest. It makes no API calls. The manifest is checked before every stage; changed settings require a new run folder.

With two repetitions per position condition, the complete study plans 400 logical calls before retries:

| Stage | Calls |
| --- | ---: |
| Two generators × 40 questions | 80 |
| Baseline judge × 80 responses | 80 |
| 40 pairs × 2 orders × 2 independent repetitions | 160 |
| Strict-rubric judge × 80 responses | 80 |
| Human review, weight-only sensitivity, analysis | 0 |

`pause_seconds` controls pacing. Quota errors stop safely after bounded retries; resume after your quota resets. Generating all outputs and judging them can span multiple days on free quotas.

## 4. Pilot the pipeline, then generate the full dataset

```bash
python -m judgebench.cli generate --run runs/first --limit 2
python -m judgebench.cli judge --run runs/first --limit 2
python -m judgebench.cli position --run runs/first --limit 4
```

Inspect saved records before scaling. `--limit` caps new derived records for that invocation; it is not a token/spend cap. Calls that were already cached do not run again.

```bash
python -m judgebench.cli generate --run runs/first
python -m judgebench.cli judge --run runs/first
python -m judgebench.cli position --run runs/first
python -m judgebench.cli judge --run runs/first --rubric strict_correctness
```

Run stages again to resume missing work. A successful raw API output is saved **before** validation. Invalid JSON, out-of-range scores, empty responses, refusals, or truncated output are not treated as valid ratings. They remain inspectable in the cache; rerunning does not silently pay for a replacement. Fix prompt/settings in a new run or explicitly inspect and remove the affected cache record before a deliberate retry. Preserve the original for your audit if you do so.

Network/provider errors are not cached as successful responses. A crash after a provider processed a request but before the cache write can still cause a duplicate call on resume; local caching cannot provide provider-side exactly-once execution.

## 5. Collect independent human ratings

```bash
streamlit run app.py
```

Select **Human review** and enter a stable reviewer ID. Responses are shuffled reproducibly for each reviewer. This page hides generator identities, automated scores, and other reviewers' annotations. The reference, checklist, and baseline rubric are visible. Scores start blank and all four must be selected.

Review all available responses, preferably with two independent reviewers. Train reviewers using only the 16 development questions; freeze guidance before evaluating the 24 held-out questions. Do not inspect automated ratings while annotating held-out responses. This is procedural blinding, not access control: the local researcher's other dashboard pages and JSON files can reveal identities and scores.

Use **Revise a saved annotation** to correct a review. The latest annotation is used for analysis and every revision is retained in `human_history/`. Different reviewers use different IDs; there is no login system. Run one analysis/export job at a time. JSON writes are atomic, but this is a local learning tool, not a concurrent multi-server annotation service.

## 6. Analyze and inspect results

```bash
python -m judgebench.cli analyze --run runs/first
```

Or choose **Recompute analysis** in the dashboard. It never makes API calls.

Pages cover agreement, calibration, position bias, rubric sensitivity, disagreements, and exports. Agreement defaults to held-out data; exploratory summaries and their denominators are also available. The dashboard labels its analysis timestamp because exports are snapshots, not automatically refreshed computations.

Results are under `runs/first/exports/`:

- `metrics.json`: coverage, per-reviewer metrics, category breakdowns, experiments, limitations.
- `findings.md`: a report computed from actual saved records; missing ratings are explicitly reported.
- CSV and JSON tables: responses, human/LLM ratings, matched ratings, disagreements and tags, calibration predictions, order-swap trials, same-order controls, rubric score deltas/rankings, human–human agreement, and API usage.
- `manifest.json`: the exact experiment configuration and dataset snapshot.

Use **Exports** to download these files or an analysis ZIP. Run folders also contain raw API payloads, prompts, request hashes, timestamps, and response identities. Preserve the entire run folder for audit; exports alone do not include every raw request. Do not publish reviewer notes without checking their contents.

## How the modules fit together

Read the code in this incremental order:

| Step | Module | Why it exists |
| --- | --- | --- |
| 1 | `schemas.py`, `dataset.py`, `rubrics.py` | Make tasks, score meanings, and valid records explicit before collecting data |
| 2 | `storage.py`, `prompts.py` | Atomic writes, content hashes, and frozen prompt versions make work reproducible |
| 3 | `providers.py`, `pipeline.py` | Keep vendor details separate from generation and experiment logic |
| 4 | `review.py`, `app.py` human-review page | Collect independent annotations on the same response identities |
| 5 | `metrics.py`, `analysis.py` | Keep statistics independently testable and export every denominator |
| 6 | Other `app.py` pages | Visualize saved data without coupling the UI to paid calls |

See [docs/METHODOLOGY.md](docs/METHODOLOGY.md) for definitions and [docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md) for a research worksheet. This project is named JudgeBench for this exercise; it does not implement or claim equivalence to any similarly named published benchmark.

## Official references

- [Groq Python quickstart](https://console.groq.com/docs/quickstart)
- [Gemini structured outputs](https://ai.google.dev/gemini-api/docs/structured-output)
- [Scikit-learn Cohen's kappa](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html)
- [Scikit-learn calibration](https://scikit-learn.org/stable/modules/calibration.html)
- [Streamlit forms](https://docs.streamlit.io/develop/api-reference/execution-flow/st.form)

