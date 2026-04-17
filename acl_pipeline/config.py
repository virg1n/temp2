from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class RuntimeConfig:
    seed: int = 42
    total_episodes: int = 100
    debug_all: bool = False
    checkpoint_every_episodes: int = 25
    log_level: str = "INFO"


@dataclass
class TaskExecutionConfig:
    enabled: bool = True
    python_executable: str = "python"
    timeout_seconds: int = 12
    max_red_generation_attempts: int = 4
    capture_max_chars: int = 1600


@dataclass
class TopicConfig:
    name: str
    weight: float = 1.0


@dataclass
class HardwareAllocation:
    gpu_ids: List[int] = field(default_factory=list)
    persistent: bool = False
    per_gpu_memory_gib: int = 46
    cpu_offload_gib: int = 128


@dataclass
class GenerationSettings:
    batch_size: int = 1
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    do_sample: bool = True
    repetition_penalty: float = 1.05


@dataclass
class LoRASettings:
    enabled: bool = True
    r: int = 64
    alpha: int = 128
    dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "down_proj",
            "gate_proj",
        ]
    )


@dataclass
class SocraticGrpoSettings:
    update_every_episodes: int = 8
    min_episodes_before_update: int = 4
    max_training_examples: int = 64
    learning_rate: float = 1e-5
    epochs: int = 1
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    beta: float = 0.02
    max_prompt_length: int = 1536
    max_completion_length: int = 256
    num_generations: int = 4
    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 2
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    full_ft: bool = False


@dataclass
class RedUpdateSettings:
    update_every_episodes: int = 12
    min_hard_examples: int = 8
    max_sft_examples: int = 256
    max_dpo_pairs: int = 128
    mining_bottom_fraction: float = 0.25
    learning_rate: float = 5e-5
    epochs: int = 1
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_length: int = 2048
    logging_steps: int = 10
    dpo_enabled: bool = True
    dpo_beta: float = 0.1


@dataclass
class RoleConfig:
    model_name_or_path: str
    tokenizer_name_or_path: Optional[str] = None
    base_adapter_path: Optional[str] = None
    quantization: Optional[str] = None
    enable_thinking: bool = False
    hardware: HardwareAllocation = field(default_factory=HardwareAllocation)
    generation: GenerationSettings = field(default_factory=GenerationSettings)
    lora: LoRASettings = field(default_factory=LoRASettings)


@dataclass
class SocraticConfig(RoleConfig):
    grpo: SocraticGrpoSettings = field(default_factory=SocraticGrpoSettings)


@dataclass
class JudgeConfig(RoleConfig):
    reward_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "no_solution_reveal": 0.30,
            "bug_localization": 0.25,
            "usefulness": 0.20,
            "socratic_style": 0.15,
            "technical_accuracy": 0.10,
        }
    )
    batch_spread_strength: float = 0.15
    episode_batch_size: int = 4


@dataclass
class RedConfig(RoleConfig):
    generation_quantization: Optional[str] = "8bit"
    update_quantization: Optional[str] = "4bit"
    update: RedUpdateSettings = field(default_factory=RedUpdateSettings)


@dataclass
class CurriculumConfig:
    topics: List[TopicConfig] = field(default_factory=list)
    reward_ema_alpha: float = 0.2
    low_reward_boost: float = 1.5
    repeat_topic_reset_threshold: int = 5


@dataclass
class StorageConfig:
    root_dir: str = "./runs/acl_pipeline"
    keep_last_n_checkpoints: int = 3
    hard_buffer_max_size: int = 2048


@dataclass
class PipelineConfig:
    runtime: RuntimeConfig
    task_execution: TaskExecutionConfig
    storage: StorageConfig
    curriculum: CurriculumConfig
    socratic: SocraticConfig
    judge: JudgeConfig
    red: RedConfig


def _read_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Expected a mapping at top level in {path}")
    return data


def _hardware(payload: Optional[Dict[str, Any]]) -> HardwareAllocation:
    payload = dict(payload or {})
    return HardwareAllocation(
        gpu_ids=[int(x) for x in payload.get("gpu_ids", [])],
        persistent=bool(payload.get("persistent", False)),
        per_gpu_memory_gib=int(payload.get("per_gpu_memory_gib", 46)),
        cpu_offload_gib=int(payload.get("cpu_offload_gib", 128)),
    )


def _generation(payload: Optional[Dict[str, Any]]) -> GenerationSettings:
    payload = dict(payload or {})
    return GenerationSettings(
        batch_size=int(payload.get("batch_size", 1)),
        max_new_tokens=int(payload.get("max_new_tokens", 256)),
        temperature=float(payload.get("temperature", 0.7)),
        top_p=float(payload.get("top_p", 0.95)),
        do_sample=bool(payload.get("do_sample", True)),
        repetition_penalty=float(payload.get("repetition_penalty", 1.05)),
    )


def _lora(payload: Optional[Dict[str, Any]]) -> LoRASettings:
    payload = dict(payload or {})
    return LoRASettings(
        enabled=bool(payload.get("enabled", True)),
        r=int(payload.get("r", 64)),
        alpha=int(payload.get("alpha", 128)),
        dropout=float(payload.get("dropout", 0.05)),
        target_modules=[str(x) for x in payload.get("target_modules", LoRASettings().target_modules)],
    )


def _socratic_grpo(payload: Optional[Dict[str, Any]]) -> SocraticGrpoSettings:
    payload = dict(payload or {})
    return SocraticGrpoSettings(
        update_every_episodes=int(payload.get("update_every_episodes", 8)),
        min_episodes_before_update=int(payload.get("min_episodes_before_update", 4)),
        max_training_examples=int(payload.get("max_training_examples", 64)),
        learning_rate=float(payload.get("learning_rate", 1e-5)),
        epochs=int(payload.get("epochs", 1)),
        per_device_batch_size=int(payload.get("per_device_batch_size", 1)),
        gradient_accumulation_steps=int(payload.get("gradient_accumulation_steps", 4)),
        warmup_ratio=float(payload.get("warmup_ratio", 0.03)),
        weight_decay=float(payload.get("weight_decay", 0.0)),
        beta=float(payload.get("beta", 0.02)),
        max_prompt_length=int(payload.get("max_prompt_length", 1536)),
        max_completion_length=int(payload.get("max_completion_length", 256)),
        num_generations=int(payload.get("num_generations", 4)),
        logging_steps=int(payload.get("logging_steps", 10)),
        save_steps=int(payload.get("save_steps", 100)),
        save_total_limit=int(payload.get("save_total_limit", 2)),
        bf16=bool(payload.get("bf16", True)),
        fp16=bool(payload.get("fp16", False)),
        gradient_checkpointing=bool(payload.get("gradient_checkpointing", True)),
        full_ft=bool(payload.get("full_ft", False)),
    )


def _red_update(payload: Optional[Dict[str, Any]]) -> RedUpdateSettings:
    payload = dict(payload or {})
    return RedUpdateSettings(
        update_every_episodes=int(payload.get("update_every_episodes", 12)),
        min_hard_examples=int(payload.get("min_hard_examples", 8)),
        max_sft_examples=int(payload.get("max_sft_examples", 256)),
        max_dpo_pairs=int(payload.get("max_dpo_pairs", 128)),
        mining_bottom_fraction=float(payload.get("mining_bottom_fraction", 0.25)),
        learning_rate=float(payload.get("learning_rate", 5e-5)),
        epochs=int(payload.get("epochs", 1)),
        per_device_batch_size=int(payload.get("per_device_batch_size", 1)),
        gradient_accumulation_steps=int(payload.get("gradient_accumulation_steps", 8)),
        max_length=int(payload.get("max_length", 2048)),
        logging_steps=int(payload.get("logging_steps", 10)),
        dpo_enabled=bool(payload.get("dpo_enabled", True)),
        dpo_beta=float(payload.get("dpo_beta", 0.1)),
    )


def _role(payload: Dict[str, Any]) -> RoleConfig:
    return RoleConfig(
        model_name_or_path=str(payload["model_name_or_path"]),
        tokenizer_name_or_path=payload.get("tokenizer_name_or_path"),
        base_adapter_path=payload.get("base_adapter_path"),
        quantization=payload.get("quantization"),
        enable_thinking=bool(payload.get("enable_thinking", False)),
        hardware=_hardware(payload.get("hardware")),
        generation=_generation(payload.get("generation")),
        lora=_lora(payload.get("lora")),
    )


def _socratic_role(payload: Dict[str, Any]) -> SocraticConfig:
    base = _role(payload)
    return SocraticConfig(
        model_name_or_path=base.model_name_or_path,
        tokenizer_name_or_path=base.tokenizer_name_or_path,
        base_adapter_path=base.base_adapter_path,
        quantization=base.quantization,
        enable_thinking=base.enable_thinking,
        hardware=base.hardware,
        generation=base.generation,
        lora=base.lora,
        grpo=_socratic_grpo(payload.get("grpo")),
    )


def _judge_role(payload: Dict[str, Any]) -> JudgeConfig:
    base = _role(payload)
    reward_weights = dict(payload.get("reward_weights") or {})
    default_weights = JudgeConfig(model_name_or_path=base.model_name_or_path).reward_weights
    return JudgeConfig(
        model_name_or_path=base.model_name_or_path,
        tokenizer_name_or_path=base.tokenizer_name_or_path,
        base_adapter_path=base.base_adapter_path,
        quantization=base.quantization,
        enable_thinking=base.enable_thinking,
        hardware=base.hardware,
        generation=base.generation,
        lora=base.lora,
        reward_weights={
            key: float(reward_weights.get(key, value))
            for key, value in default_weights.items()
        },
        batch_spread_strength=float(payload.get("batch_spread_strength", 0.15)),
        episode_batch_size=int(payload.get("episode_batch_size", 4)),
    )


def _red_role(payload: Dict[str, Any]) -> RedConfig:
    base = _role(payload)
    return RedConfig(
        model_name_or_path=base.model_name_or_path,
        tokenizer_name_or_path=base.tokenizer_name_or_path,
        base_adapter_path=base.base_adapter_path,
        quantization=base.quantization,
        enable_thinking=base.enable_thinking,
        hardware=base.hardware,
        generation=base.generation,
        lora=base.lora,
        generation_quantization=payload.get("generation_quantization", "8bit"),
        update_quantization=payload.get("update_quantization", "4bit"),
        update=_red_update(payload.get("update")),
    )


def load_config(path: str, *, debug_all_override: Optional[bool] = None) -> PipelineConfig:
    raw = _read_yaml(Path(path))

    runtime_raw = dict(raw.get("runtime") or {})
    task_execution_raw = dict(raw.get("task_execution") or {})
    storage_raw = dict(raw.get("storage") or {})
    curriculum_raw = dict(raw.get("curriculum") or {})
    topics_raw = curriculum_raw.get("topics") or []
    if not topics_raw:
        raise ValueError("Config must define curriculum.topics")

    runtime = RuntimeConfig(
        seed=int(runtime_raw.get("seed", 42)),
        total_episodes=int(runtime_raw.get("total_episodes", 100)),
        debug_all=bool(runtime_raw.get("debug_all", False)),
        checkpoint_every_episodes=int(runtime_raw.get("checkpoint_every_episodes", 25)),
        log_level=str(runtime_raw.get("log_level", "INFO")).upper(),
    )
    if debug_all_override:
        runtime.debug_all = True

    task_execution = TaskExecutionConfig(
        enabled=bool(task_execution_raw.get("enabled", True)),
        python_executable=str(task_execution_raw.get("python_executable", "python")),
        timeout_seconds=int(task_execution_raw.get("timeout_seconds", 12)),
        max_red_generation_attempts=int(task_execution_raw.get("max_red_generation_attempts", 4)),
        capture_max_chars=int(task_execution_raw.get("capture_max_chars", 1600)),
    )

    storage = StorageConfig(
        root_dir=str(storage_raw.get("root_dir", "./runs/acl_pipeline")),
        keep_last_n_checkpoints=int(storage_raw.get("keep_last_n_checkpoints", 3)),
        hard_buffer_max_size=int(storage_raw.get("hard_buffer_max_size", 2048)),
    )

    curriculum = CurriculumConfig(
        topics=[
            TopicConfig(name=str(item["name"]), weight=float(item.get("weight", 1.0)))
            for item in topics_raw
        ],
        reward_ema_alpha=float(curriculum_raw.get("reward_ema_alpha", 0.2)),
        low_reward_boost=float(curriculum_raw.get("low_reward_boost", 1.5)),
        repeat_topic_reset_threshold=int(curriculum_raw.get("repeat_topic_reset_threshold", 5)),
    )

    return PipelineConfig(
        runtime=runtime,
        task_execution=task_execution,
        storage=storage,
        curriculum=curriculum,
        socratic=_socratic_role(dict(raw["socratic"])),
        judge=_judge_role(dict(raw["judge"])),
        red=_red_role(dict(raw["red"])),
    )
