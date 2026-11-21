# Research worksheet

**Status: not run.** Complete this worksheet with observed evidence after collecting real evaluations. No example study numbers are supplied.

## Before collecting test data

- Run folder and manifest ID:
- Model IDs, provider access, account quotas, date checked:
- Dataset/reference review completed by:
- Reviewer IDs and training procedure:
- Freeze date for prompts, rubric, binary rule, and test partition:
- Known threats to independence or blinding:

## Coverage

Record response, judge, human, complete order-pair, dev/test, and category counts. Explain missing observations and whether quota failures selected the evaluated subset.

## Human vs. judge

Record per-reviewer held-out exact agreement, kappa, weighted kappa, signed error, and binary confusion matrix/precision/recall/F1. Compare with human–human agreement. Note undefined metrics or severe class imbalance. Cite exported files and sample sizes.

## Position bias

Compare swapped-order flip rate, outright reversals, tie transitions, and first-position selection with same-order variability. Inspect inconsistent examples. Explain why the available evidence does or does not support a position-effect interpretation.

## Rubric sensitivity

Separate changed judge criteria from changed aggregation weights. Inspect the same candidate pairs; describe tie changes separately from outright reversals. Identify whether the intended quality construct changed.

## Calibration

Record dev/test labeled counts, bin counts, raw Brier, and dev-fitted calibrated Brier when eligible. Check for class imbalance and overfitting. Do not select a calibrator on the test results.

## Qualitative findings and limitations

For recurring disagreements, record question/response IDs, evidence, initial issue tags, and any later human adjudication. Include counterexamples. State what this small study cannot establish and which experiment should run next.
