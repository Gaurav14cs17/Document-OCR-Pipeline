from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Dict

import torch
from PIL import Image

from vlm_pipeline.core.config import VALID_TASKS

TaskHandler = Callable[[Image.Image], Dict[str, Any]]


class VLMBackend(ABC):
    """Base class for all vision-language model backends."""

    name: str
    model_id: str = "unknown"
    device: str = "cpu"
    quant_mode: str = "auto"

    @abstractmethod
    def run_task(
        self,
        image: Image.Image,
        task: str,
        max_new_tokens: int | None = None,
    ) -> Dict[str, Any]: ...

    @classmethod
    def resolve_device(cls, device: str | None = None) -> str:
        return device or ("cuda" if torch.cuda.is_available() else "cpu")

    def task_result(self, task: str, **payload: Any) -> Dict[str, Any]:
        return {"task": task, "backend": self.name, **payload}

    def validate_task(self, task: str) -> None:
        if task not in VALID_TASKS:
            raise ValueError(f"Unsupported task '{task}' for backend '{self.name}'")

    def _dispatch(
        self,
        image: Image.Image,
        task: str,
        handlers: Dict[str, TaskHandler],
    ) -> Dict[str, Any]:
        self.validate_task(task)
        if task not in handlers:
            raise ValueError(f"Task '{task}' is not implemented by backend '{self.name}'")
        return handlers[task](image)

    def run_dispatch(
        self,
        image: Image.Image,
        task: str,
        handlers: Dict[str, TaskHandler],
        max_new_tokens: int | None = None,
    ) -> Dict[str, Any]:
        """Run a task via a handler map. Ignores max_new_tokens unless handlers use it."""
        return self._dispatch(image, task, handlers)
