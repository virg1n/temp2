"""Adversarial curriculum training pipeline for Socratic Python tutoring."""

from .config import PipelineConfig, load_config

__all__ = ["AdversarialCurriculumPipeline", "PipelineConfig", "load_config"]


def __getattr__(name: str):
    if name == "AdversarialCurriculumPipeline":
        from .pipeline import AdversarialCurriculumPipeline

        return AdversarialCurriculumPipeline
    raise AttributeError(name)
