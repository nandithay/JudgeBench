"""Validate data at boundaries, before it enters metrics or the dashboard."""
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

DIMENSIONS = ("correctness", "relevance", "completeness", "instruction_following")
Score = Annotated[int, Field(strict=True, ge=1, le=5)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Question(StrictModel):
    id: str
    category: Literal["factual_qa", "reasoning", "summarization", "instruction_following"]
    split: Literal["dev", "test"]
    prompt: str = Field(min_length=1)
    reference: str = Field(min_length=1)
    checklist: list[str] = Field(min_length=1)
    provenance: str = Field(min_length=1)


class Scores(StrictModel):
    correctness: Score
    relevance: Score
    completeness: Score
    instruction_following: Score

    def good(self) -> bool:
        """Preregistered positive class: every dimension scores at least four."""
        return min(self.model_dump().values()) >= 4


class Reasons(StrictModel):
    correctness: str = Field(min_length=1)
    relevance: str = Field(min_length=1)
    completeness: str = Field(min_length=1)
    instruction_following: str = Field(min_length=1)


class JudgeRating(StrictModel):
    scores: Scores
    reasons: Reasons
    p_good: float = Field(ge=0, le=1, allow_inf_nan=False)


class PairRating(StrictModel):
    winner: Literal["A", "B", "tie"]
    reason: str = Field(min_length=1)


ERROR_TAGS = ["factual_error", "reasoning_error", "omission", "instruction_violation",
              "verbosity_preference", "unsupported_claim", "rubric_ambiguity",
              "reference_issue", "other"]


class HumanRating(StrictModel):
    response_id: str
    reviewer: str = Field(min_length=1, max_length=80, pattern=r"^[a-zA-Z0-9_-]+$")
    scores: Scores
    notes: str = ""
    tags: list[Literal["factual_error", "reasoning_error", "omission", "instruction_violation",
                       "verbosity_preference", "unsupported_claim", "rubric_ambiguity",
                       "reference_issue", "other"]] = Field(default_factory=list)
    rubric_id: Literal["baseline"] = "baseline"


class ModelSpec(StrictModel):
    provider: Literal["groq", "gemini"]
    model: str = Field(min_length=1)
    temperature: float = Field(ge=0, le=2)
    max_tokens: int = Field(ge=128, le=32768)


class Settings(StrictModel):
    generators: list[ModelSpec] = Field(min_length=2, max_length=2)
    judge: ModelSpec
    seed: int = 2026
    position_repeats: int = Field(default=2, ge=2, le=10)
    pause_seconds: float = Field(default=6, ge=0, le=60)
    max_attempts: int = Field(default=3, ge=1, le=5)

    @model_validator(mode="after")
    def different_models(self):
        names = [s.model for s in self.generators] + [self.judge.model]
        if len(set(names)) != 3:
            raise ValueError("Use three distinct model IDs, including across providers.")
        return self
