from __future__ import annotations

import json
import re

from .config import ACTConfig
from .hard_rules import apply_hard_rules
from .openai_client import ChatRequest, OpenAIChatClient
from .prompts import format_judge_messages
from .schemas import JudgeResult


def _extract_json(text: str) -> dict:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.IGNORECASE).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(stripped[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Judge response must be a JSON object.")
    return value


class SocraticJudge:
    def __init__(self, client: OpenAIChatClient, cfg: ACTConfig) -> None:
        self.client = client
        self.cfg = cfg

    def score(self, code: str, error: str, hint: str) -> JudgeResult:
        raw_response = self.client.complete(
            ChatRequest(
                messages=format_judge_messages(code, error, hint),
                temperature=self.cfg.generation.judge_temperature,
                top_p=self.cfg.generation.judge_top_p,
                max_tokens=self.cfg.generation.judge_max_tokens,
            )
        )
        try:
            judge_json = _extract_json(raw_response)
            raw_score = float(judge_json.get("score", 0.0))
            reason = str(judge_json.get("reason", ""))
        except Exception as exc:
            judge_json = {"parse_error": str(exc), "raw_response": raw_response}
            raw_score = 0.0
            reason = f"Judge JSON parse failure: {exc}"

        adjusted = apply_hard_rules(hint, code, raw_score, judge_json, self.cfg.judge)
        return JudgeResult(
            raw_score=max(0.0, min(10.0, raw_score)),
            final_score=adjusted.score,
            reason=reason,
            flags=adjusted.flags,
            hard_rule_flagged=bool(adjusted.flags),
            score_ceiling=adjusted.score_ceiling,
            judge_json=judge_json,
        )
