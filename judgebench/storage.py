"""Atomic local JSON writes and content-addressed request caching."""
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    """A crash must not leave half of a result file behind."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def records(folder):
    return [read_json(p) for p in sorted(Path(folder).glob("*.json"))]
