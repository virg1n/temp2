from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    return repr(value)


class StructuredLogger:
    def __init__(self, root_dir: str, *, level: str = "INFO", debug_all: bool = False) -> None:
        self.debug_all = debug_all
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("acl_pipeline")
        self.logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        self.logger.handlers.clear()
        self.logger.propagate = False

        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)

        file_handler = logging.FileHandler(self.root_dir / "run.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def event(self, event_name: str, **payload: Any) -> None:
        self.logger.info("%s | %s", event_name, json.dumps(payload, ensure_ascii=False, default=_json_default))

    def warning(self, event_name: str, **payload: Any) -> None:
        self.logger.warning("%s | %s", event_name, json.dumps(payload, ensure_ascii=False, default=_json_default))

    def error(self, event_name: str, **payload: Any) -> None:
        self.logger.error("%s | %s", event_name, json.dumps(payload, ensure_ascii=False, default=_json_default))

    def debug_dump(self, event_name: str, **payload: Any) -> None:
        if self.debug_all:
            self.event(event_name, **payload)


def build_logger(root_dir: str, *, level: str = "INFO", debug_all: bool = False) -> StructuredLogger:
    return StructuredLogger(root_dir=root_dir, level=level, debug_all=debug_all)
