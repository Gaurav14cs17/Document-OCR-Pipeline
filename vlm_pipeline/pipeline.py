from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image

from vlm_pipeline.backends import get_backend
from vlm_pipeline.core.config import FAST_PRESETS, OVERLAY_TASKS, TASK_DPI
from vlm_pipeline.utils.image import resize_long_edge
from vlm_pipeline.utils.io import load_pages
from vlm_pipeline.utils.visualize import save_overlays, save_page_comparison


class DocumentVLMPipeline:
    """Document OCR pipeline: one VLM backend, multiple prompt-routed tasks."""

    TASK_DPI = TASK_DPI
    FAST_PRESETS = FAST_PRESETS

    def __init__(
        self,
        backend: str = "qwen",
        model_id: str | None = None,
        quant: str = "auto",
        fast: bool = False,
        dpi: int | None = None,
        max_image_size: int | None = None,
        max_tokens: int | None = None,
        models_dir: str | None = "models",
    ):
        backend_cls = get_backend(backend)
        kwargs: dict[str, Any] = {"quant": quant, "models_dir": models_dir}
        if model_id:
            kwargs["model_id"] = model_id
        self.backend = backend_cls(**kwargs)
        self.models_dir = models_dir
        self.quant = quant
        self.fast = fast
        self.dpi = dpi or (FAST_PRESETS["dpi"] if fast else None)
        self.max_image_size = max_image_size or (
            FAST_PRESETS["max_image_size"] if fast else 1024
        )
        self.max_tokens = max_tokens or (
            FAST_PRESETS["max_tokens"] if fast else 2048
        )
        self.default_tasks = list(FAST_PRESETS["tasks"] if fast else ["ocr"])

    def run(
        self,
        input_path: str | Path,
        tasks: List[str] | None = None,
        page_range: List[int] | None = None,
        output_dir: str | Path = "vlm_output",
        save_images: bool = True,
    ) -> Dict[str, Any]:
        if tasks is None:
            tasks = list(self.default_tasks)
        for task in tasks:
            if task not in TASK_DPI:
                raise ValueError(f"Unknown task '{task}'")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        pages_out: List[Dict[str, Any]] = []
        for page_idx in page_range or [0]:
            page_bundle: Dict[str, Any] = {"page_index": page_idx, "tasks": {}}
            overlays: List[str] = []

            for task in tasks:
                dpi = self.dpi or TASK_DPI[task]
                image = load_pages(input_path, dpi=dpi, page_range=[page_idx])[0]
                original_size = image.size
                image = resize_long_edge(image, self.max_image_size)
                task_result = self.backend.run_task(
                    image,
                    task,
                    max_new_tokens=self.max_tokens,
                )
                task_result["image_size"] = {
                    "original": list(original_size),
                    "model_input": list(image.size),
                    "dpi": dpi,
                    "max_image_size": self.max_image_size,
                }
                page_bundle["tasks"][task] = task_result

                if save_images and task in OVERLAY_TASKS:
                    overlays.extend(
                        save_overlays(
                            image,
                            task_result,
                            output_path / f"page_{page_idx}",
                            page_idx,
                        )
                    )

            if save_images:
                compare_image = load_pages(input_path, dpi=96, page_range=[page_idx])[0]
                compare_image = resize_long_edge(compare_image, self.max_image_size)
                comparison_path = save_page_comparison(
                    compare_image,
                    page_bundle["tasks"],
                    output_path / f"page_{page_idx}",
                    page_idx,
                )
                if comparison_path:
                    overlays.append(comparison_path)

            page_bundle["overlay_images"] = overlays
            pages_out.append(page_bundle)

        result = {
            "input": str(input_path),
            "backend": self.backend.name,
            "model": getattr(self.backend, "model_id", "unknown"),
            "quant": getattr(self.backend, "quant_mode", self.quant),
            "device": getattr(self.backend, "device", "unknown"),
            "fast": self.fast,
            "max_image_size": self.max_image_size,
            "max_tokens": self.max_tokens,
            "dpi": self.dpi,
            "pages": pages_out,
        }

        results_file = output_path / "results.json"
        with results_file.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        result["results_file"] = str(results_file)
        return result
