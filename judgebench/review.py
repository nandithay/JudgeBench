"""One independent baseline annotation per response per reviewer, with revision history."""
from pathlib import Path
from uuid import uuid4

from .pipeline import manifest_for
from .schemas import HumanRating
from .storage import now, records, write_json


def human_records(run):
    paths = sorted((Path(run) / "human").glob("*/*.json"))
    from .storage import read_json
    return [read_json(path) for path in paths]


def save_review(run, annotation):
    manifest = manifest_for(run)
    annotation = HumanRating.model_validate(annotation)
    response_ids = {r["response_id"] for r in records(Path(run) / "responses")}
    if annotation.response_id not in response_ids:
        raise ValueError("Unknown response: annotation cannot be matched to this run.")
    record = {**annotation.model_dump(), "created_at": now(),
              "experiment_id": manifest["experiment_id"]}
    # Keep a separate immutable revision before replacing the current annotation.
    write_json(Path(run) / "human_history" / f"{uuid4().hex}.json", record)
    write_json(Path(run) / "human" / annotation.reviewer / f"{annotation.response_id}.json", record)
    return record
