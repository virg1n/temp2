from __future__ import annotations

from dataclasses import replace

from logging_utils import get_logger, log_event
from models_factory import TransformersChatModel
from schemas import PipelineSettings, PipelineState


LOGGER = get_logger(__name__)


class EnvironmentEngine:
    def __init__(self, settings: PipelineSettings, state: PipelineState) -> None:
        self.settings = settings
        self.state = state
        self._judge: TransformersChatModel | None = None
        self._red: TransformersChatModel | None = None
        self._socratic: TransformersChatModel | None = None
        self._red_signature: tuple[str, str | None] | None = None
        self._socratic_signature: tuple[str, str | None] | None = None

    def load_judge(self) -> TransformersChatModel:
        if self._judge is None:
            self._judge = TransformersChatModel(self.settings.models.judge)
            self._judge.load()
        return self._judge

    def load_red(self, *, use_base_model: bool = False) -> TransformersChatModel:
        adapter_path = None if use_base_model else self.state.red.active_adapter_path or self.settings.models.red.adapter_path
        signature = (self.settings.models.red.model_name_or_path, adapter_path)
        if self._red is None or self._red_signature != signature:
            self.unload_red()
            config = replace(self.settings.models.red, adapter_path=adapter_path)
            self._red = TransformersChatModel(config, adapter_override=adapter_path)
            self._red.load()
            self._red_signature = signature
            log_event(LOGGER, 20, "red_ready", "Red runtime prepared", use_base_model=use_base_model, adapter_path=adapter_path)
        return self._red

    def load_socratic(self) -> TransformersChatModel:
        model_path = self.state.socratic.active_model_path or self.settings.models.socratic.model_name_or_path
        adapter_path = self.state.socratic.active_adapter_path or self.settings.models.socratic.adapter_path
        signature = (model_path, adapter_path)
        if self._socratic is None or self._socratic_signature != signature:
            self.unload_socratic()
            config = replace(self.settings.models.socratic, model_name_or_path=model_path, adapter_path=adapter_path)
            self._socratic = TransformersChatModel(config, adapter_override=adapter_path)
            self._socratic.load()
            self._socratic_signature = signature
            log_event(LOGGER, 20, "socratic_ready", "Socratic runtime prepared", model_path=model_path, adapter_path=adapter_path)
        return self._socratic

    def unload_red(self) -> None:
        if self._red is not None:
            self._red.unload(empty_cuda_cache=self.settings.runtime.empty_cuda_cache_on_unload)
        self._red = None
        self._red_signature = None

    def unload_socratic(self) -> None:
        if self._socratic is not None:
            self._socratic.unload(empty_cuda_cache=self.settings.runtime.empty_cuda_cache_on_unload)
        self._socratic = None
        self._socratic_signature = None

    def close(self) -> None:
        self.unload_red()
        self.unload_socratic()
        if self._judge is not None:
            self._judge.unload(empty_cuda_cache=self.settings.runtime.empty_cuda_cache_on_unload)
        self._judge = None
