"""Stdlib-only checks, runnable even before third-party dependencies are installed."""
import ast
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from judgebench.dataset import build_dataset
from judgebench.storage import digest, read_json, write_json


class OfflineChecks(unittest.TestCase):
    def test_authored_dataset_matches_json(self):
        rows = build_dataset(validate=False)
        self.assertEqual(rows, read_json("data/questions.json"))
        self.assertEqual(len(rows), 40)
        self.assertEqual(len({r["id"] for r in rows}), 40)
        self.assertEqual(set(Counter(r["category"] for r in rows).values()), {10})
        self.assertEqual(Counter(r["split"] for r in rows), {"dev": 16, "test": 24})

    def test_references_and_checklists_are_present(self):
        for row in build_dataset(validate=False):
            self.assertTrue(all(row[k] for k in ("prompt", "reference", "checklist", "provenance")))
        fictional = [r for r in build_dataset(validate=False) if r["category"] == "summarization"]
        self.assertTrue(all("fictional" in r["provenance"] for r in fictional))

    def test_reference_constraint_examples(self):
        rows = {r["id"]: r for r in build_dataset(validate=False)}
        self.assertEqual(len(rows["instruction_following-10"]["reference"].split()), 6)
        self.assertEqual(json.loads(rows["instruction_following-09"]["reference"]), [2, 4, 6, 8, 10])
        self.assertEqual(len(rows["instruction_following-08"]["reference"].splitlines()), 3)

    def test_atomic_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "test.json"
            write_json(path, {"value": "é", "items": [1, 2]})
            write_json(path, {"value": "updated"})
            self.assertEqual(read_json(path), {"value": "updated"})
            self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_nonfinite_values_rejected(self):
        with self.assertRaises(ValueError):
            digest({"value": float("nan")})

    def test_request_hash_canonical_and_trial_sensitive(self):
        self.assertEqual(digest({"b": 2, "a": 1}), digest({"a": 1, "b": 2}))
        self.assertNotEqual(digest({"trial": 0}), digest({"trial": 1}))

    def test_python_sources_parse(self):
        paths = [Path("app.py")] + list(Path("judgebench").glob("*.py")) + list(Path("tests").glob("*.py"))
        for path in paths:
            with self.subTest(path=str(path)):
                ast.parse(path.read_text(), filename=str(path))


if __name__ == "__main__":
    unittest.main()
