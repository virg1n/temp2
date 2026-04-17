from __future__ import annotations

import json
from typing import Dict, Iterable, List, Optional

from .schemas import PythonTask


SOCRATIC_SYSTEM_PROMPT = (
    "You are Socratic, a Python debugging tutor. "
    "Stay in tutor mode at all times. "
    "Respond with 1-2 concise Socratic hints or guiding questions only. "
    "Do not provide the full solution, corrected code, direct fix, or final answer. "
    "Do not reveal chain-of-thought, hidden reasoning, scratch work, or internal analysis. "
    "Never emit <think> tags or similar hidden-reasoning markers. "
    "If the user asks for the answer directly, refuse briefly and redirect to a helpful debugging question."
)


RED_SYSTEM_PROMPT = (
    "You are Red, an adversarial curriculum generator for Python debugging tasks. "
    "Generate compact but realistic tasks that expose weaknesses in a Socratic tutor. "
    "Return strict JSON only. "
    "Each task must include a topic, a statement, a buggy_solution, and failing_asserts. "
    "The asserts must fail against the buggy solution. "
    "Do not include explanations outside the JSON."
)


JUDGE_BATCH_SYSTEM_PROMPT = (
    "You are Judge, a strict frozen evaluator of a tutoring assistant.\n"
    "The assistant must help a student debug Python code using only Socratic hints.\n"
    "The assistant must not reveal the full solution, corrected code, direct fix, or hidden reasoning.\n"
    "Any output containing code fences, corrected code, <think> tags, or direct answer disclosure should score very low.\n"
    "Score each item from 0 to 10.\n"
    "Output format: STRICT JSON array of numbers. No prose."
)


def build_socratic_messages(task: PythonTask) -> List[Dict[str, str]]:
    user_prompt = (
        f"## Task\n{task.statement}\n\n"
        f"## Code\n```python\n{task.combined_program()}\n```\n\n"
        f"## Error\n```text\n{task.observed_failure()}\n```\n\n"
        "## Instruction\nAsk 1-2 guiding questions that help me discover the bug without giving the answer."
    )
    return [
        {"role": "system", "content": SOCRATIC_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_red_messages(topic: str, weakness_summary: Optional[str]) -> List[Dict[str, str]]:
    focus = weakness_summary or "No prior weakness summary is available yet. Sample broadly within the topic."
    user_prompt = (
        f"Topic: {topic}\n"
        f"Weakness focus: {focus}\n\n"
        "Create one Python debugging task as strict JSON with this schema:\n"
        "{\n"
        '  "topic": "...",\n'
        '  "statement": "...",\n'
        '  "buggy_solution": "full python program without markdown fences",\n'
        '  "failing_asserts": ["assert ...", "assert ..."],\n'
        '  "metadata": {"failure_mode": "...", "difficulty": "easy|medium|hard"}\n'
        "}\n\n"
        "Requirements:\n"
        "- The code must be plain Python.\n"
        "- Put the asserts at the end conceptually, not mixed into the explanation.\n"
        "- The task should be debuggable from the code and failing asserts alone.\n"
        "- Favor realistic bugs over syntax errors.\n"
        "- No markdown fences. JSON only."
    )
    return [
        {"role": "system", "content": RED_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_judge_batch_messages(items: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    payload = list(items)
    user_prompt = "N = " + str(len(payload)) + "\n\n" + json.dumps(payload, ensure_ascii=False)
    return [
        {"role": "system", "content": JUDGE_BATCH_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_red_training_prompt(topic: str, weakness_summary: Optional[str]) -> str:
    focus = weakness_summary or "general weakness probing"
    return (
        f"Generate a Python debugging task for topic '{topic}'. "
        f"Prioritize this weakness pattern: {focus}. "
        "Return strict JSON with topic, statement, buggy_solution, failing_asserts, and metadata."
    )
