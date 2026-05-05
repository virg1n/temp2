from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TopicConfig:
    name: str
    base_weight: float = 1.0
    notes: str = ""


@dataclass
class RunConfig:
    output_dir: str = "runs/act_qwen3"
    seed: int = 17
    language: str = "python"
    max_iterations: int = 500


@dataclass
class ModelConfig:
    socratic_base_model: str = "Qwen/Qwen3-1.7B"
    socratic_adapter: str | None = None
    red_model: str = "Qwen/Qwen3-32B"
    judge_model: str = "Qwen/Qwen3-32B"
    vllm_base_url: str = "http://127.0.0.1:8000/v1"
    vllm_api_key: str = "EMPTY"


@dataclass
class GenerationConfig:
    red_candidates_per_topic: int = 3
    red_max_tokens: int = 2600
    red_temperature: float = 0.85
    red_top_p: float = 0.9
    jailbreak_probability: float = 0.08
    socratic_candidates: int = 4
    socratic_max_new_tokens: int = 150
    socratic_temperature: float = 0.8
    socratic_top_p: float = 0.9
    judge_max_tokens: int = 600
    judge_temperature: float = 0.0
    judge_top_p: float = 1.0


@dataclass
class CurriculumConfig:
    ema_alpha: float = 0.2
    initial_ema_reward: float = 0.7
    low_reward_boost_lambda: float = 2.0
    max_same_topic_streak: int = 3
    min_score_gap: float = 1.5
    hard_reward_max: float = 6.5
    topics: list[TopicConfig] = field(default_factory=lambda: [
        TopicConfig("lists and indexing", 1.0, "off-by-one, wrong index, empty list edge case"),
        TopicConfig("loops and accumulators", 1.0, "wrong initialization, update in wrong branch"),
        TopicConfig("conditionals", 1.0, "boundary condition or inverted comparison"),
        TopicConfig("strings", 0.9, "case handling, slicing, counting characters"),
        TopicConfig("dictionaries", 0.9, "missing key handling, wrong value update"),
        TopicConfig("sorting", 0.8, "wrong key, ascending versus descending, tie break"),
        TopicConfig("functions and return values", 0.8, "forgotten return, returning too early"),
        TopicConfig("nested loops", 0.7, "wrong loop variable or reset location"),
        TopicConfig("basic recursion", 0.5, "base case or recursive argument, keep it short"),
    ])


@dataclass
class JudgeConfig:
    max_hint_words: int = 120
    hard_max_hint_words: int = 160
    min_alphabetic_words: int = 8
    max_questions: int = 2
    no_question_penalty: float = 1.5
    no_question_score_ceiling: float = 2.0
    low_alpha_words_score_ceiling: float = 2.0
    punctuation_markdown_zero_threshold: float = 0.55
    markdown_zero_threshold: float = 0.30
    too_many_questions_penalty: float = 1.0
    too_long_penalty: float = 2.0
    code_block_penalty: float = 3.0
    generic_penalty: float = 1.5
    direct_fix_forced_score: float = 0.0
    leakage_multiplier: float = 0.1


@dataclass
class ExecutionConfig:
    python_executable: str = "python"
    timeout_seconds: float = 2.0
    max_code_chars: int = 6000
    allowed_imports: list[str] = field(default_factory=lambda: [
        "collections",
        "functools",
        "itertools",
        "math",
        "random",
        "statistics",
    ])


@dataclass
class SocraticDPOConfig:
    beta: float = 0.1
    learning_rate: float = 5.0e-6
    epochs: int = 1
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    max_length: int = 4096
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    load_in_4bit: bool = False


@dataclass
class RedKPOConfig:
    beta: float = 0.08
    reward_temperature: float = 0.7
    learning_rate: float = 2.0e-6
    epochs: int = 1
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_length: int = 6144
    max_candidates_per_group: int = 4
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    load_in_4bit: bool = True


@dataclass
class TrainConfig:
    socratic_dpo: SocraticDPOConfig = field(default_factory=SocraticDPOConfig)
    red_kpo: RedKPOConfig = field(default_factory=RedKPOConfig)


@dataclass
class ACTConfig:
    run: RunConfig = field(default_factory=RunConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def _deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _as_dict(config: ACTConfig) -> dict[str, Any]:
    return {
        "run": vars(config.run),
        "models": vars(config.models),
        "generation": vars(config.generation),
        "curriculum": {
            **{k: v for k, v in vars(config.curriculum).items() if k != "topics"},
            "topics": [vars(topic) for topic in config.curriculum.topics],
        },
        "judge": vars(config.judge),
        "execution": vars(config.execution),
        "train": {
            "socratic_dpo": vars(config.train.socratic_dpo),
            "red_kpo": vars(config.train.red_kpo),
        },
    }


def config_from_mapping(data: dict[str, Any]) -> ACTConfig:
    merged = _deep_update(_as_dict(ACTConfig()), data)
    return ACTConfig(
        run=RunConfig(**merged["run"]),
        models=ModelConfig(**merged["models"]),
        generation=GenerationConfig(**merged["generation"]),
        curriculum=CurriculumConfig(
            **{k: v for k, v in merged["curriculum"].items() if k != "topics"},
            topics=[TopicConfig(**topic) for topic in merged["curriculum"]["topics"]],
        ),
        judge=JudgeConfig(**merged["judge"]),
        execution=ExecutionConfig(**merged["execution"]),
        train=TrainConfig(
            socratic_dpo=SocraticDPOConfig(**merged["train"]["socratic_dpo"]),
            red_kpo=RedKPOConfig(**merged["train"]["red_kpo"]),
        ),
    )


def load_config(path: str | Path) -> ACTConfig:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load ACT YAML configs.") from exc

    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping, got {type(data)!r}")
    return config_from_mapping(data)
