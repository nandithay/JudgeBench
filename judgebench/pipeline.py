"""Freeze a run, then generate, score, and swap response pairs in resumable stages."""
import importlib.metadata
import json
import random
import sys
import time
from pathlib import Path

from .dataset import load_dataset
from .prompts import PROMPTS
from .providers import call_api
from .rubrics import all_rubrics
from .schemas import JudgeRating, PairRating, Settings
from .storage import digest, now, read_json, records, write_json


def init_run(run, config="config.json", dataset="data/questions.json"):
    run = Path(run)
    if run.exists() and any(run.iterdir()):
        raise ValueError("Run folder is not empty. Choose a new folder to preserve the old experiment.")
    settings = Settings.model_validate(read_json(config)).model_dump()
    packages = {}
    for name in ["judgebench-local", "pandas", "numpy", "scipy", "pydantic", "scikit-learn",
                 "matplotlib", "groq", "google-genai", "streamlit", "pytest"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not installed"
    manifest = {"format_version": 1, "created_at": now(), "config": settings,
                "questions": load_dataset(dataset), "rubrics": all_rubrics(),
                "prompts": PROMPTS, "packages": packages, "python_version": sys.version,
                "quality_rule": "all four baseline scores >= 4",
                "dataset_note": "Hand-authored learning dataset. Fictional source passages are explicitly labeled."}
    manifest["experiment_id"] = digest(manifest)
    write_json(run / "manifest.json", manifest)
    return manifest


def manifest_for(run):
    manifest = read_json(Path(run) / "manifest.json")
    expected = manifest["experiment_id"]
    if digest({k: v for k, v in manifest.items() if k != "experiment_id"}) != expected:
        raise ValueError("Run manifest was modified. Initialize a new run instead.")
    return manifest


def request(run, spec, system, payload, trial, settings, schema=None, caller=None):
    """Trial is part of the key: independent repetitions must make independent calls."""
    body = {"model": spec, "system": system,
            "user": json.dumps(payload, sort_keys=True, ensure_ascii=False),
            "schema": schema, "trial": trial}
    key = digest(body)
    path = Path(run) / "cache" / f"{key}.json"
    if path.exists():
        result = read_json(path)
    else:
        if settings["pause_seconds"] and caller is None:
            time.sleep(settings["pause_seconds"])
        result = (caller or call_api)(spec, system, body["user"], schema=schema,
                                      max_attempts=settings["max_attempts"])
        result = {**result, "request": body, "request_id": key, "created_at": now()}
        write_json(path, result)  # Persist BEFORE parsing so malformed output is not paid for twice.
    if not result.get("complete") or not result.get("text", "").strip():
        raise ValueError(f"Empty, blocked, or truncated output saved in cache/{key}.json. "
                         "Inspect it; create a new run with adjusted settings rather than grading it.")
    return result


def _context(q):
    return {k: q[k] for k in ("prompt", "reference", "checklist")}


def generate(run, limit=None, caller=None):
    manifest = manifest_for(run)
    settings = manifest["config"]
    completed = 0
    for q in manifest["questions"]:
        for index, spec in enumerate(settings["generators"]):
            rid = digest({"question": q["id"], "spec": spec, "experiment": manifest["experiment_id"]})
            path = Path(run) / "responses" / f"{rid}.json"
            if path.exists():
                continue
            if limit is not None and completed >= limit:
                return completed
            # References/checklists are NEVER passed to the response generators.
            result = request(run, spec, manifest["prompts"]["generator"],
                             {"task": q["prompt"]}, "generation-0", settings, caller=caller)
            write_json(path, {"response_id": rid, "question_id": q["id"],
                              "generator_index": index, "model": spec,
                              "text": result["text"], "request_id": result["request_id"],
                              "created_at": result["created_at"]})
            completed += 1
    return completed


def judge(run, rubric_id="baseline", limit=None, caller=None):
    manifest = manifest_for(run)
    settings = manifest["config"]
    rubric = manifest["rubrics"][rubric_id]
    questions = {q["id"]: q for q in manifest["questions"]}
    completed = 0
    for response in records(Path(run) / "responses"):
        path = Path(run) / "judgments" / rubric_id / f"{response['response_id']}.json"
        if path.exists():
            continue
        if limit is not None and completed >= limit:
            break
        payload = {**_context(questions[response["question_id"]]), "candidate": response["text"],
                   "rubric": rubric}
        result = request(run, settings["judge"], manifest["prompts"]["judge"], payload,
                         f"score-{rubric_id}-0", settings, JudgeRating.model_json_schema(), caller)
        rating = JudgeRating.model_validate_json(result["text"])
        write_json(path, {"response_id": response["response_id"], "question_id": response["question_id"],
                          "rubric_id": rubric_id, **rating.model_dump(),
                          "request_id": result["request_id"], "created_at": result["created_at"]})
        completed += 1
    return completed


def position(run, limit=None, caller=None):
    manifest = manifest_for(run)
    settings = manifest["config"]
    by_question = {}
    for response in records(Path(run) / "responses"):
        by_question.setdefault(response["question_id"], {})[response["generator_index"]] = response
    jobs = []
    for q in manifest["questions"]:
        pair = by_question.get(q["id"], {})
        if len(pair) != 2:
            continue  # Missing responses are not treated as ties or failures.
        for repeat in range(settings["position_repeats"]):
            for order in ["AB", "BA"]:
                jobs.append((q, pair, repeat, order))
    # Interleave order and repeat conditions to reduce confounding with time.
    random.Random(settings["seed"]).shuffle(jobs)
    completed = 0
    for q, pair, repeat, order in jobs:
        path = Path(run) / "position" / f"{q['id']}-{repeat}-{order}.json"
        if path.exists():
            continue
        if limit is not None and completed >= limit:
            break
        first, second = (pair[0], pair[1]) if order == "AB" else (pair[1], pair[0])
        payload = {**_context(q), "rubric": manifest["rubrics"]["baseline"],
                   "A": first["text"], "B": second["text"]}
        result = request(run, settings["judge"], manifest["prompts"]["pair"], payload,
                         f"position-{repeat}-{order}", settings, PairRating.model_json_schema(), caller)
        rating = PairRating.model_validate_json(result["text"])
        mapping = {"A": first["response_id"], "B": second["response_id"], "tie": "tie"}
        write_json(path, {"question_id": q["id"], "repeat": repeat, "order": order,
                          "first_response_id": first["response_id"], "second_response_id": second["response_id"],
                          "selected_position": rating.winner, "winner_response_id": mapping[rating.winner],
                          "reason": rating.reason, "request_id": result["request_id"]})
        completed += 1
    return completed
