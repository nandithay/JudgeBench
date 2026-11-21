# Validation status

Build date: 2026-11-19.

## Checks actually executed

- Seven standard-library offline checks passed: dataset/file identity and balance; required reference/checklist fields; constrained reference examples; atomic storage round trip; nonfinite hash rejection; canonical/trial-sensitive request hashes; Python source parsing.
- `compileall` completed successfully for the application, modules, and tests with Python 3.10.19.
- Neither `GROQ_API_KEY` nor `GEMINI_API_KEY` was present in the build environment. No live LLM generation/judging was run, and no human scores were manufactured.

Offline checks can be repeated without third-party packages:

```bash
python -m unittest discover -s tests -p test_offline_core.py -v
python -m compileall -q judgebench app.py tests
```

## Checks implemented but not executed here

The full `pytest` suite includes schema validation, known-answer agreement/classification metrics, undefined kappa, cluster resampling, dev/test calibration isolation, order-identity mapping, ties/incomplete trials, caching/resume, independent repetition keys, invalid-output retention, provider error redaction/retries, annotation revisions, rubric ranking changes, a partial offline pipeline integration, and Streamlit's headless empty-state check.

These tests could not be executed in this environment: dependency installation could not reach the package index, required libraries were not preinstalled, and the available package-manager fallback failed. This is an environment limitation, not a passing test result. A working Python 3.10+ installation and internet access to install packages are required to validate the full application.

After installing dependencies, run:

```bash
python -m pytest -q
```

Then configure keys and run a two-response pilot before a full study. Provider integration behavior, live model availability, account quota, and the rendered dashboard remain unverified here. Do not infer experimental findings or production readiness from the seven offline checks alone.
