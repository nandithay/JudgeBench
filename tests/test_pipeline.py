"""All model outputs in this file are synthetic test fixtures, NOT study results."""
import json
import pytest

from judgebench.analysis import analyze, joined_ratings, sensitivity
from judgebench.dataset import build_dataset
from judgebench.pipeline import generate, init_run, judge, manifest_for, position
from judgebench.providers import APIError, call_api
from judgebench.review import human_records, save_review
from judgebench.schemas import DIMENSIONS
from judgebench.storage import read_json, records, write_json


@pytest.fixture
def run(tmp_path):
    config = read_json("config.json")
    config["pause_seconds"] = 0
    write_json(tmp_path / "config.json", config)
    write_json(tmp_path / "questions.json", build_dataset())
    path = tmp_path / "test_run"
    init_run(path, tmp_path / "config.json", tmp_path / "questions.json")
    return path


def fake(spec, system, user, schema=None, max_attempts=3):
    if schema and "winner" in schema.get("properties", {}):
        text = json.dumps({"winner": "A", "reason": "SYNTHETIC fixture: first item chosen."})
    elif schema:
        text = json.dumps({"scores": dict.fromkeys(DIMENSIONS, 4),
                           "reasons": dict.fromkeys(DIMENSIONS, "SYNTHETIC evidence"), "p_good": 0.6})
    else:
        # Ensure generators receive no reference answers/checklists.
        assert set(json.loads(user)) == {"task"}
        text = "SYNTHETIC TEST RESPONSE"
    return {"text": text, "complete": True, "raw": {"synthetic_test_fixture": True}, "latency_seconds": 0, "attempts": 1}


def test_resume_does_not_repeat_completed_calls(run):
    calls = []
    def caller(*args, **kwargs):
        calls.append(1)
        return fake(*args, **kwargs)
    assert generate(run, 2, caller) == 2
    assert len(calls) == 2
    # Remove derived output, leaving raw cache; recovery still makes no API calls.
    for path in (run / "responses").glob("*.json"):
        path.unlink()
    assert generate(run, 2, caller) == 2
    assert len(calls) == 2


def test_repeated_position_trials_have_different_requests(run):
    generate(run, 2, fake)
    assert position(run, caller=fake) == 4
    rows = records(run / "position")
    assert len({r["request_id"] for r in rows}) == 4
    assert {r["winner_response_id"] for r in rows} == {r["response_id"] for r in records(run / "responses")}
    assert position(run, caller=fake) == 0


def test_malformed_output_saved_before_validation(run):
    generate(run, 1, fake)
    calls = []
    def malformed(*args, **kwargs):
        calls.append(1)
        return {"text": '{"scores":{}}', "complete": True, "raw": {}}
    with pytest.raises(ValueError):
        judge(run, caller=malformed)
    with pytest.raises(ValueError):
        judge(run, caller=malformed)
    assert len(calls) == 1
    assert len(records(run / "judgments" / "baseline")) == 0


def test_failed_requests_are_not_cached_as_success(run):
    def failing(*args, **kwargs):
        raise APIError("synthetic failure")
    with pytest.raises(APIError):
        generate(run, 1, failing)
    assert not records(run / "cache")
    assert generate(run, 1, fake) == 1


def test_immutable_manifest(run):
    manifest = read_json(run / "manifest.json")
    manifest["config"]["seed"] = 42
    write_json(run / "manifest.json", manifest)
    with pytest.raises(ValueError):
        manifest_for(run)
    with pytest.raises(ValueError):
        init_run(run)


def test_end_to_end_partial_real_format_no_fabricated_humans(run):
    generate(run, 2, fake)
    judge(run, caller=fake)
    judge(run, "strict_correctness", caller=fake)
    position(run, caller=fake)
    report = analyze(run)
    assert report["coverage"]["human_annotations"] == 0
    assert report["reviewers"] == {}
    assert report["position"]["flip_rate"] == 1
    assert (run / "exports" / "findings.md").exists()
    assert read_json(run / "exports" / "matched_ratings.json") == []
    rid = records(run / "responses")[0]["response_id"]
    annotation = dict(response_id=rid, reviewer="tester", scores=dict.fromkeys(DIMENSIONS, 4))
    save_review(run, annotation)
    assert len(joined_ratings(run)) == 1
    result = analyze(run)
    assert result["reviewers"]["tester"]["quality"]["f1"] == 1
    annotation["scores"]["correctness"] = 2
    save_review(run, annotation)
    assert len(human_records(run)) == 1
    assert len(records(run / "human_history")) == 2
    assert len(read_json(run / "exports" / "human_ratings.json")) == 1
    with pytest.raises(ValueError):
        save_review(run, {**annotation, "response_id": "unknown"})


def test_sensitivity_matching_and_ranking_changes(run):
    generate(run, 2, fake)
    judge(run, caller=fake)
    judge(run, "strict_correctness", caller=fake)
    strict_path = sorted((run / "judgments" / "strict_correctness").glob("*.json"))[0]
    strict = read_json(strict_path)
    strict["scores"]["correctness"] = 1
    write_json(strict_path, strict)
    summary, deltas, rankings = sensitivity(run)
    assert summary["matched_responses"] == 2
    assert summary["mean_score_delta"] == -0.375
    assert summary["conditions"]["criteria_only"]["ranking_change_rate"] == 1
    assert summary["conditions"]["criteria_only"]["outright_reversal_rate"] == 0
    assert summary["conditions"]["weights_only"]["ranking_change_rate"] == 0


def test_human_human_does_not_require_automated_scores(run):
    generate(run, 1, fake)
    rid = records(run / "responses")[0]["response_id"]
    for reviewer in ["r1", "r2"]:
        save_review(run, dict(response_id=rid, reviewer=reviewer, scores=dict.fromkeys(DIMENSIONS, 4)))
    summary = analyze(run)
    assert summary["coverage"]["matched_annotations"] == 0
    assert len(summary["human_human"]) == 4
    assert all(r["n"] == 1 and r["exact_agreement"] == 1 for r in summary["human_human"])


def test_provider_does_not_expose_secret_error_text(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "local-test-secret")
    def fail(*args, **kwargs):
        raise RuntimeError("local-test-secret must never appear")
    monkeypatch.setattr("judgebench.providers._call", fail)
    with pytest.raises(APIError) as error:
        call_api(read_json("config.json")["generators"][0], "system", "user")
    assert "local-test-secret" not in str(error.value)


def test_transient_retry_is_bounded(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake")
    calls = []
    class Limited(Exception):
        status_code = 429
    def flaky(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise Limited()
        return {"text": "fixture", "raw": {}, "complete": True}
    monkeypatch.setattr("judgebench.providers._call", flaky)
    monkeypatch.setattr("judgebench.providers.time.sleep", lambda _: None)
    result = call_api(read_json("config.json")["generators"][0], "system", "user", max_attempts=2)
    assert len(calls) == 2 and result["attempts"] == 2
