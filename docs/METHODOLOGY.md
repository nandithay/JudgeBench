# Methodology and interpretation

## Study design

The evaluation unit is a **specific saved response to a question**. Two distinct models answer each of 40 hand-authored questions. The judge sees the task, response, reference, checklist, and scoring rubric, but not generator identities. Generators see only the task, never the reference/checklist. A separate pairwise prompt is used for position bias.

Each category has 10 questions: factual QA, reasoning, summarization, instruction following. The first 4 in each category are development questions and the remaining 6 are test questions: 16 dev / 24 test, or 32 / 48 candidate responses. These are fixed partitions, not a random claim of population representativeness. Summarization passages are explicitly fictional. Reference answers are authored guides; alternatives may be valid and reference problems can be tagged by humans. Review reference correctness before collecting labels.

Freeze models, prompts, task wording, rubric, and analysis definitions before test evaluation. If pilot results drive changes, initialize a new run. Do not tune on the test partition. Prefer pinned model snapshots when a provider exposes them; aliases can change server-side even when the saved string is unchanged.

## Rubric and binary quality

All four dimensions are scored independently on integer 1–5 scales with dimension-specific anchors in `rubrics.py`: correctness, relevance, completeness, instruction following. Human review and baseline LLM scoring use exactly the same anchors, checklist, and scope guidance. Scores are ordinal; means and MAE are useful descriptive summaries, not evidence of equal psychological spacing.

The positive quality label is preregistered as **every baseline dimension >=4**. A strong relevance score cannot hide a factual failure. Binary precision, recall, and F1 use human classifications as the reference and judge classifications as predictions. Confusion matrices have human labels on rows, judge labels on columns, with order `[not_good, good]`. Precision/recall undefined by zero denominators are reported as zero with explicit flags; do not interpret those zeros as well-estimated performance.

## Human vs. automated judge

Join ratings using response IDs, not text similarity or row order. Report each reviewer's agreement separately. Never concatenate multiple reviewers and treat their correlated annotations as independent observations. No automatic consensus is invented. Human–human Cohen's kappa provides an essential context for judge–human disagreement.

Per dimension report:

- Exact agreement = matched identical scores / matched ratings.
- Cohen's kappa = `(observed agreement − expected agreement) / (1 − expected agreement)`.
- Quadratic weighted kappa penalizes larger ordinal gaps more heavily.
- Mean absolute error and signed judge-minus-human difference describe magnitude and leniency.

Constant identical ratings can have perfect exact agreement but undefined kappa (expected agreement is also one). Undefined kappa is `null`, never replaced by zero or one. Small class diversity and prevalence can produce surprising kappa values; inspect distributions and confusion matrices, not a single headline.

Question-cluster percentile bootstrap intervals use 500 draws and a fixed seed for exact and binary agreement. Both generated responses remain together when resampling their question. Position flip intervals likewise cluster repeated trials by question. Intervals are unavailable with fewer than two question clusters. These exploratory intervals do not solve dataset selection bias, human rater bias, multiple comparisons, or provider drift. Kappa/F1 point estimates do not currently have bootstrap intervals.

Held-out agreement and position summaries are explicitly separated from overall exploratory summaries. Category metrics retain split labels and sample counts. Missing human or judge labels are excluded from paired calculations, not silently treated as zero scores. Coverage counts expose how much evidence is available.

## Position bias

For each question, freeze the same two saved candidate texts. Judge both display orders, AB and BA, using identical evaluation instructions. Use two independent calls for **each** order by default. A seeded shuffle interleaves question, order, and repetition to reduce timing confounds. Repeated calls have distinct cache keys even at temperature zero.

Map displayed winners back to response identity before comparing. A consistent judge can select displayed A in AB and displayed B in BA. Report:

- Preference consistency: identical identity or tie across the two orders / complete order pairs.
- Flip rate: any difference in identity/tie outcome / complete order pairs.
- Outright reversal: different non-tie winners / complete order pairs.
- Tie transition: exactly one order gives a tie / complete order pairs.
- First-position selection over all judgments, and separately over non-tie judgments.
- Tie rate and all relevant denominators.
- Same-order change rate across adjacent independent repetitions.

Only complete order pairs enter consistency/reversal calculations. Incomplete order trials are still visible as records and count toward raw judgment coverage. First-position selection is easier to interpret after both orders have completed. An always-tie judge has high consistency but little discriminative value. Swapped-order differences are descriptive and can arise from stochasticity, time-varying serving, prompt wording, or position effects. Same-order controls help diagnose this, but we do not subtract their rates and call the result a causal bias estimate.

## Rubric sensitivity

Two separate interventions:

1. **Criteria-only:** rejudge each response using the strict-correctness variant. Unsupported substantive claims or invalid reasoning cap correctness at 2. The other anchors and equal weights remain the same. Report within-response dimension/mean deltas, and within-question ranking changes using matched candidate pairs only.
2. **Weights-only:** keep baseline scores fixed and use weights `{correctness: .55, relevance: .10, completeness: .10, instruction_following: .25}` instead of `.25` each. No API calls. Report ranking changes and outright reversals separately; changing to/from a tie counts as a ranking change, not an outright reversal.

Ranking refers to which of the two responses is preferred within a question, not a universal model leaderboard. A tiny, easy dataset can yield mostly ties. Sensitivity summaries are exploratory across all available matched questions and include per-question exports; they are not held-out evidence of improvement. The criteria experiment has one judgment per response per rubric: rubric effects remain confounded with normal judge variability. Repeat rubric trials in a later extension if stronger causal evidence is needed.

## Calibration

The judge returns `p_good`: its probability that an independent trained human using the **baseline** rubric will assign all four dimensions >=4. This is not a normalized score or confidence that the judge's own answer is right. The event definition stays fixed under the rubric experiment.

For each reviewer, compute raw Brier score `mean((p_good − human_good)^2)` on **held-out test** responses. Five equal-width probability bins show mean forecast vs. observed human-good fraction, with bin counts. Expected calibration error is a descriptive weighted absolute bin gap. Calibration at probability 1 belongs to the final bin.

Fit an isotonic regressor only if at least 10 labeled dev responses, both human classes, and at least two distinct predicted probabilities exist. Apply it to untouched test probabilities and report raw vs. calibrated Brier scores on the same test responses. Save test predictions for inspection. With insufficient dev labels, show only raw held-out calibration; do not create a fitted curve or claim improvement.

This is a teaching demonstration: a small dev set can overfit isotonic regression, and lower Brier loss reflects both calibration and discrimination. A one-class test set makes findings particularly weak. Self-reported LLM confidence is not guaranteed to be calibrated. Plot bin sample sizes and consider an expanded dataset before deployment decisions.

## Disagreement/error analysis

Flag a response-reviewer pair when any absolute dimension gap is >=2 or binary quality differs. Inspect task, reference, candidate, scores, and judge reasons. Human reviewers can independently tag factual errors, reasoning errors, omissions, instruction violations, verbosity preference, unsupported claims, rubric ambiguity, reference issues, or other issues.

Tag counts describe recurring **annotated response issues among disagreements**, not automatically proven judge mistakes. Distinguish a judge error, human annotation error, and ambiguous reference/rubric through manual adjudication. The first version intentionally does not auto-label error causes using the same judge being tested. Agreement alone cannot establish correctness.

## Reproducibility and boundaries

Manifest hashes cover the frozen config, dataset, rubric, prompts, installed package versions, and creation metadata. Cache hashes cover model settings, full prompts, output schema, and repetition identity. Raw outputs include provider metadata and usage; validated records link back to cache request IDs. Each record is saved atomically. There are no silent substitutions, automatic synthetic study data, or automatic paid-model fallbacks.

This is not provider-side exactly-once execution or cryptographic tamper-proof auditing. The manifest is integrity-checked, but manually editing response/judgment files is outside the supported workflow. Stable config does not force deterministic server behavior. Original API usage/cost, quota schedules, model retirement, provider data-use policies, multilingual tasks, adversarial judge injection, self-preference, verbosity bias, long-context tasks, and general benchmark validity require additional investigation.

