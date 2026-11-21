from collections import Counter

import pytest
from pydantic import ValidationError

from judgebench.dataset import build_dataset, validate_dataset
from judgebench.rubrics import all_rubrics
from judgebench.schemas import DIMENSIONS, HumanRating, JudgeRating, Scores, Settings
from judgebench.storage import read_json


def test_dataset_is_balanced_and_frozen_file_matches():
    rows = build_dataset()
    assert len(rows) == 40
    assert set(Counter(q["category"] for q in rows).values()) == {10}
    assert Counter(q["split"] for q in rows) == {"dev": 16, "test": 24}
    assert rows == read_json("data/questions.json")
    for row in rows:
        assert row["reference"] and row["checklist"] and row["provenance"]


@pytest.mark.parametrize("value", [0, 6, 3.5, "4", True])
def test_score_rejects_invalid_types_and_ranges(value):
    with pytest.raises(ValidationError):
        Scores(**dict.fromkeys(DIMENSIONS, value))


def test_good_requires_every_dimension():
    assert Scores(**dict.fromkeys(DIMENSIONS, 4)).good()
    assert not Scores(**{**dict.fromkeys(DIMENSIONS, 5), "correctness": 3}).good()


def test_duplicate_questions_rejected():
    rows = build_dataset()
    rows[-1] = rows[0]
    with pytest.raises(ValueError):
        validate_dataset(rows)


def test_reviewer_path_traversal_rejected():
    with pytest.raises(ValidationError):
        HumanRating(response_id="x", reviewer="../secret", scores=dict.fromkeys(DIMENSIONS, 4))


def test_distinct_model_requirement():
    cfg = read_json("config.json")
    cfg["judge"] = cfg["generators"][0]
    with pytest.raises(ValidationError):
        Settings(**cfg)


def test_rubric_has_every_anchor():
    for rubric in all_rubrics().values():
        assert set(rubric["anchors"]) == set(DIMENSIONS)
        assert all(set(anchors) == {"1", "2", "3", "4", "5"} for anchors in rubric["anchors"].values())


@pytest.mark.parametrize("probability", [-0.1, 1.1, float("nan")])
def test_bad_probabilities_rejected(probability):
    with pytest.raises(ValidationError):
        JudgeRating(scores=dict.fromkeys(DIMENSIONS, 4), reasons=dict.fromkeys(DIMENSIONS, "evidence"), p_good=probability)
