from __future__ import annotations

from typing import Type

from vlm_pipeline.backends.base import VLMBackend
from vlm_pipeline.backends.florence2 import Florence2Backend
from vlm_pipeline.backends.got_ocr2 import GotOcr2Backend
from vlm_pipeline.backends.qwen import QwenVLBackend

BACKEND_REGISTRY: dict[str, Type[VLMBackend]] = {
    "florence2": Florence2Backend,
    "qwen": QwenVLBackend,
    "gotocr2": GotOcr2Backend,
}

BACKENDS = BACKEND_REGISTRY


def get_backend(name: str) -> Type[VLMBackend]:
    if name not in BACKEND_REGISTRY:
        raise ValueError(f"Unknown backend '{name}'. Choose from: {list(BACKEND_REGISTRY)}")
    return BACKEND_REGISTRY[name]
