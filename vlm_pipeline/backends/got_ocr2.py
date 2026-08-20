from __future__ import annotations

from typing import Any, Dict, List

import torch
from PIL import Image

from vlm_pipeline.backends.base import VLMBackend


class GotOcr2Backend(VLMBackend):
    """Dedicated open OCR VLM (~580M). Runs in-process, no server."""

    name = "gotocr2"

    def __init__(
        self,
        model_id: str = "stepfun-ai/GOT-OCR2_0",
        device: str | None = None,
        quant: str = "auto",
    ):
        from transformers import AutoModel, AutoTokenizer

        self.model_id = model_id
        self.device = self.resolve_device(device)
        if self.device != "cuda":
            raise RuntimeError(
                "GOT-OCR2 requires an NVIDIA GPU (upstream model uses CUDA internally). "
                "Use --backend qwen for CPU/GPU flexible inference."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            device_map="auto",
        )
        self.model.eval()

    def _save_temp(self, image: Image.Image) -> str:
        path = "/tmp/vlm_pipeline_page.png"
        image.save(path)
        return path

    def _chat(self, image: Image.Image, ocr_type: str = "ocr") -> str:
        return self.model.chat(self.tokenizer, self._save_temp(image), ocr_type=ocr_type)

    def _handlers(self) -> dict[str, Any]:
        return {
            "ocr": self._got_ocr,
            "table": self._got_table,
            "detect": self._got_detect,
            "layout": self._got_layout,
        }

    def run_task(self, image: Image.Image, task: str, max_new_tokens: int | None = None) -> Dict[str, Any]:
        return self._dispatch(image, task, self._handlers())

    def _got_ocr(self, image: Image.Image) -> Dict[str, Any]:
        text = self._chat(image, ocr_type="ocr")
        return self.task_result("ocr", text=text, html=f"<pre>{text}</pre>")

    def _got_table(self, image: Image.Image) -> Dict[str, Any]:
        text = self._chat(image, ocr_type="format")
        return self.task_result(
            "table",
            tables=[{"raw": text}],
            note="GOT-OCR2 format mode output",
        )

    def _got_detect(self, image: Image.Image) -> Dict[str, Any]:
        text = self._chat(image, ocr_type="format")
        return self.task_result(
            "detect",
            text_lines=[{"text": text, "bbox": None, "confidence": 1.0}],
            note="GOT-OCR2 returns formatted text; use qwen backend for bbox JSON.",
        )

    def _got_layout(self, image: Image.Image) -> Dict[str, Any]:
        text = self._chat(image, ocr_type="format")
        blocks: List[Dict[str, Any]] = [
            {"label": "Text", "bbox": None, "reading_order": i, "caption": line[:200]}
            for i, line in enumerate((ln for ln in text.splitlines() if ln.strip()), start=1)
        ]
        return self.task_result("layout", blocks=blocks)