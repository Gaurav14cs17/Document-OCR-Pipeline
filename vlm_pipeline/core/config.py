from __future__ import annotations

TASK_DPI = {
    "layout": 96,
    "detect": 96,
    "ocr": 192,
    "table": 192,
}

FAST_PRESETS = {
    "dpi": 96,
    "max_image_size": 1024,
    "max_tokens": 1536,
    "tasks": ["ocr"],
}

VALID_TASKS = frozenset(TASK_DPI.keys())
OVERLAY_TASKS = frozenset({"detect", "layout"})
