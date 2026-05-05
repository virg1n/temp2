from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RedTask:
    topic: str
    statement: str
    reference_solution: str
    buggy_solution: str
    intended_bug: str
    asserts: list[str]
    difficulty: str = "easy"
    tags: list[str] = field(default_factory=list)
    raw_completion: str = ""
    prompt_text: str = ""
    messages: list[dict[str, str]] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationResult:
    valid: bool
    error_message: str = ""
    rejection_reasons: list[str] = field(default_factory=list)
    reference_stdout: str = ""
    reference_stderr: str = ""
    buggy_stdout: str = ""
    buggy_stderr: str = ""

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class JudgeResult:
    raw_score: float
    final_score: float
    reason: str
    flags: list[str] = field(default_factory=list)
    hard_rule_flagged: bool = False
    score_ceiling: float | None = None
    judge_json: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HintRollout:
    prompt: str
    hint: str
    judge: JudgeResult
    topic: str
    task_id: str

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["judge"] = self.judge.to_json()
        return data
