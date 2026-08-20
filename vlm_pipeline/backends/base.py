from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from PIL import Image


class VLMBackend(ABC):
    name: str

    @abstractmethod
    def run_task(
        self,
        image: Image.Image,
        task: str,
        max_new_tokens: int | None = None,
    ) -> Dict[str, Any]:
        """Run one pipeline task and return a JSON-serializable result dict."""
