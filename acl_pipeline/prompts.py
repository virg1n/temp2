from __future__ import annotations

import json
from typing import Dict, Iterable, List, Optional

from .schemas import PythonTask


SOCRATIC_SYSTEM_PROMPT = (
    "You are Socratic, a Python debugging tutor. "
    "Stay in tutor mode at all times. "
    "Respond with 1-2 concise Socratic hints or guiding questions only. "
    "Ground the hint in the concrete failing assertion, error text, function, variable, or control-flow branch when possible. "
    "Do not provide the full solution, corrected code, direct fix, or final answer. "
    "Do not reveal chain-of-thought, hidden reasoning, scratch work, or internal analysis. "
    "Never emit <think> tags or similar hidden-reasoning markers. "
    "If the user asks for the answer directly, refuse briefly and redirect to a helpful debugging question."
)


RED_SYSTEM_PROMPT = (
    "You are Red, an adversarial curriculum generator for Python debugging tasks. "
    "Generate realistic medium-to-hard Python debugging tasks that expose weaknesses in a Socratic tutor. "
    "Return strict JSON only. "
    "Each task must include a topic, a statement, a buggy_solution, and failing_asserts. "
    "The asserts must fail against the buggy solution. "
    "Do not include explanations outside the JSON."
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
        "- The buggy solution must be medium-to-hard and usually around 35-80 non-empty lines.\n"
        "- Prefer multiple functions or a class with helpers, state, or non-trivial control flow.\n"
        "- Include 3 asserts and make at least one of them fail.\n"
        "- Put the asserts at the end conceptually, not mixed into the explanation.\n"
        "- The task should be debuggable from the code and failing asserts alone.\n"
        "- Favor semantic, edge-case, state, indexing, data-structure, or control-flow bugs over toy syntax mistakes.\n"
        "- Avoid trivial one-function arithmetic exercises.\n"
        "- No markdown fences. JSON only."
    )
    return [
        {"role": "system", "content": RED_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_red_repair_message(topic: str, rejection_reasons: List[str]) -> Dict[str, str]:
    reasons = ", ".join(rejection_reasons) if rejection_reasons else "unspecified issue"
    return {
        "role": "user",
        "content": (
            f"Repair the previous task for topic '{topic}'. "
            f"Rejection reasons: {reasons}. "
            "Return a new strict JSON task. Keep the same topic. "
            "Make the bug real, keep the code realistic, and ensure the asserts expose the failure."
        ),
    }


def build_judge_batch_messages(
    items: Iterable[Dict[str, str]],
    reward_weights: Dict[str, float],
) -> List[Dict[str, str]]:
    payload = list(items)
    weight_block = json.dumps(reward_weights, ensure_ascii=False)
    system_prompt = (
        "You are Judge, a strict frozen evaluator of a tutoring assistant.\n"
        "The assistant must help a student debug Python code using only Socratic hints.\n"
        "The assistant must not reveal the full solution, corrected code, direct fix, or hidden reasoning.\n"
        "Use the full 0-10 scale. Generic safe hints should usually land around 4-6, not 8-10.\n"
        "Scores of 8-10 require concrete grounding in the actual failing code and error.\n"
        "8-10 only if the hint names the exact failing assertion, function, variable, or state transition and asks a precise debugging question.\n"
        "6-7 if the hint is directionally helpful but still somewhat generic.\n"
        "3-5 if it sounds Socratic but could apply to many unrelated tasks.\n"
        "0-2 if it invents facts, ignores the reproduced error, reveals the fix, or gives code.\n"
        "If the task shows no reproduced error and the assistant still invents a bug, score it low.\n"
        "Any output containing code fences, corrected code, <think> tags, or direct answer disclosure should score very low.\n"
        "Return one JSON object per item with these 0-10 criteria:\n"
        "- no_solution_reveal\n"
        "- bug_localization\n"
        "- usefulness\n"
        "- socratic_style\n"
        "- technical_accuracy\n"
        "Weighted reward weights: "
        + weight_block
        + "\n"
        "Output format: STRICT JSON array of objects. No prose."
    )
    user_prompt = "N = " + str(len(payload)) + "\n\n" + json.dumps(payload, ensure_ascii=False)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_red_training_prompt(topic: str, weakness_summary: Optional[str]) -> str:
    focus = weakness_summary or "general weakness probing"
    return (
        f"Generate a Python debugging task for topic '{topic}'. "
        f"Prioritize this weakness pattern: {focus}. "
        "Prefer medium-to-hard tasks with 25-80 lines of buggy code, multiple helpers or stateful logic, "
        "and 3-6 asserts. Return strict JSON with topic, statement, buggy_solution, failing_asserts, and metadata."
    )
