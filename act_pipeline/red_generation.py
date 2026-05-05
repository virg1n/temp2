from __future__ import annotations

import json
import random
import re
from typing import Any

from .config import ACTConfig, TopicConfig
from .openai_client import ChatRequest, OpenAIChatClient
from .prompts import format_red_messages, render_messages_plain
from .schemas import RedTask


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


def coerce_red_task(data: dict[str, Any], raw: str, prompt_text: str, messages: list[dict[str, str]]) -> RedTask:
    required = ["topic", "statement", "reference_solution", "buggy_solution", "intended_bug", "asserts"]
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"Red output missing required keys: {', '.join(missing)}")

    asserts = data["asserts"]
    if isinstance(asserts, str):
        asserts = [line.strip() for line in asserts.splitlines() if line.strip()]
    if not isinstance(asserts, list) or not all(isinstance(line, str) for line in asserts):
        raise ValueError("Red output field 'asserts' must be a list of strings.")

    tags = data.get("tags", [])
    if isinstance(tags, str):
        tags = [tags]
    if not isinstance(tags, list):
        tags = []

    return RedTask(
        topic=str(data["topic"]),
        statement=str(data["statement"]),
        reference_solution=str(data["reference_solution"]),
        buggy_solution=str(data["buggy_solution"]),
        intended_bug=str(data["intended_bug"]),
        asserts=[line.strip() for line in asserts if line.strip()],
        difficulty=str(data.get("difficulty", "easy")),
        tags=[str(tag) for tag in tags],
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
        "completion": raw,
        "valid": True,
        "score": 0.0,
        "rejection_reasons": [],
    }
