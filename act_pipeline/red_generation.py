from __future__ import annotations

import io
import json
import random
import re
import tokenize
from typing import Any

from .config import ACTConfig, TopicConfig
from .openai_client import ChatRequest, OpenAIChatClient
from .prompts import format_red_messages, render_messages_plain
from .schemas import RedTask


MIN_SOLUTION_CODE_LINES = 25


def extract_json_object(text: str) -> dict[str, Any]:
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
        raise ValueError("Red output JSON must be an object.")
    return value


def strip_python_comments(code: str) -> str:
    """Remove Python comment tokens while preserving strings and code layout."""
    tokens: list[tokenize.TokenInfo] = []
    try:
        for token in tokenize.generate_tokens(io.StringIO(code).readline):
            if token.type == tokenize.COMMENT:
                continue
            tokens.append(token)
        cleaned = tokenize.untokenize(tokens)
    except tokenize.TokenError:
        cleaned = code
    lines = [line.rstrip() for line in cleaned.splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def _code_line_count(code: str) -> int:
    return sum(1 for line in code.splitlines() if line.strip())


def _normalize_asserts(asserts: list[str]) -> list[str]:
    cleaned = strip_python_comments("\n".join(asserts))
    return [line.strip() for line in cleaned.splitlines() if line.strip()]


def _canonical_completion(task: RedTask) -> str:
    payload = {
        "topic": task.topic,
        "statement": task.statement,
        "reference_solution": task.reference_solution,
        "buggy_solution": task.buggy_solution,
        "asserts": task.asserts,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def extract_code_completion(text: str) -> str:
    stripped = text.strip()
    fenced = re.search(r"```(?:python)?\s*(.*?)```", stripped, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        stripped = fenced.group(1).strip()
    return strip_python_comments(stripped)


def coerce_red_task(data: dict[str, Any], raw: str, prompt_text: str, messages: list[dict[str, str]]) -> RedTask:
    required = ["topic", "statement", "reference_solution", "buggy_solution", "asserts"]
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"Red output missing required keys: {', '.join(missing)}")

    asserts = data["asserts"]
    if isinstance(asserts, str):
        asserts = [line.strip() for line in asserts.splitlines() if line.strip()]
    if not isinstance(asserts, list) or not all(isinstance(line, str) for line in asserts):
        raise ValueError("Red output field 'asserts' must be a list of strings.")

    reference_solution = strip_python_comments(str(data["reference_solution"]))
    buggy_solution = strip_python_comments(str(data["buggy_solution"]))
    normalized_asserts = _normalize_asserts([line for line in asserts if line.strip()])

    too_short = []
    if _code_line_count(reference_solution) < MIN_SOLUTION_CODE_LINES:
        too_short.append("reference_solution")
    if _code_line_count(buggy_solution) < MIN_SOLUTION_CODE_LINES:
        too_short.append("buggy_solution")
    if too_short:
        raise ValueError(
            f"{', '.join(too_short)} must contain at least {MIN_SOLUTION_CODE_LINES} non-empty code lines"
        )

    return RedTask(
        topic=str(data["topic"]),
        statement=str(data["statement"]),
        reference_solution=reference_solution,
        buggy_solution=buggy_solution,
        asserts=normalized_asserts,
        raw_completion=raw,
        prompt_text=prompt_text,
        messages=messages,
    )


def build_red_prompt(topic: TopicConfig, rng: random.Random, cfg: ACTConfig) -> tuple[list[dict[str, str]], str]:
    include_jailbreak = rng.random() < cfg.generation.jailbreak_probability
    messages = format_red_messages(topic, include_jailbreak=include_jailbreak)
    return messages, render_messages_plain(messages)


def generate_red_candidate(
    client: OpenAIChatClient,
    topic: TopicConfig,
    rng: random.Random,
    cfg: ACTConfig,
) -> tuple[RedTask | None, dict[str, Any]]:
    messages, prompt_text = build_red_prompt(topic, rng, cfg)
    raw = client.complete(
        ChatRequest(
            messages=messages,
            temperature=cfg.generation.red_temperature,
            top_p=cfg.generation.red_top_p,
            max_tokens=cfg.generation.red_max_tokens,
            seed=rng.randint(0, 2**31 - 1),
        )
    )
    try:
        data = extract_json_object(raw)
        task = coerce_red_task(data, raw, prompt_text, messages)
    except Exception as exc:
        return None, {
            "prompt": prompt_text,
            "messages": messages,
            "completion": raw,
            "valid": False,
            "score": 0.0,
            "rejection_reasons": [f"JSON/schema error: {exc}"],
        }
    return task, {
        "prompt": prompt_text,
        "messages": messages,
        "raw_completion": raw,
        "completion": _canonical_completion(task),
        "valid": True,
        "score": 0.0,
        "rejection_reasons": [],
    }


def repair_buggy_solution(
    client: OpenAIChatClient,
    task: RedTask,
    rng: random.Random,
    cfg: ACTConfig,
) -> tuple[str | None, dict[str, Any]]:
    repair_messages = [
        *task.messages,
        {"role": "assistant", "content": task.raw_completion},
        {
            "role": "user",
            "content": (
                "The previous buggy_solution passed all asserts, so it is already correct. "
                "Change only buggy_solution so at least one assert fails. "
                "Output only the full replacement buggy Python code, not JSON. "
                "Do not output markdown, comments, docstrings, or explanation. "
                f"Keep the same public function signatures and at least {MIN_SOLUTION_CODE_LINES} non-empty code lines."
            ),
        },
    ]
    raw = client.complete(
        ChatRequest(
            messages=repair_messages,
            temperature=cfg.generation.red_temperature,
            top_p=cfg.generation.red_top_p,
            max_tokens=cfg.generation.red_max_tokens,
            seed=rng.randint(0, 2**31 - 1),
        )
    )
    code = extract_code_completion(raw)
    record = {
        "messages": repair_messages,
        "raw_completion": raw,
        "completion": code,
        "valid": bool(code.strip()),
        "rejection_reasons": [],
    }
    if _code_line_count(code) < MIN_SOLUTION_CODE_LINES:
        record["valid"] = False
        record["rejection_reasons"] = [
            f"repair buggy_solution must contain at least {MIN_SOLUTION_CODE_LINES} non-empty code lines"
        ]
        return None, record
    return code, record


def canonical_red_completion(task: RedTask) -> str:
    return _canonical_completion(task)
