from __future__ import annotations

from typing import Any, Dict, List
import torch
from PIL import Image
from vlm_pipeline.backends.base import VLMBackend


class GotOcr2Backend(VLMBackend):
    """Dedicated open OCR VLM (~580M). Runs in-process, no server."""

    name = "gotocr2"

    def __init__(self, model_id: str = "stepfun-ai/GOT-OCR2_0", device: str | None = None, quant: str = "auto"):
        from transformers import AutoModel, AutoTokenizer

        self.model_id = model_id
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        if device != "cuda":
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
        image_path = self._save_temp(image)
        return self.model.chat(self.tokenizer, image_path, ocr_type=ocr_type)

    def run_task(
        self,
        image: Image.Image,
        task: str,
        max_new_tokens: int | None = None,
    ) -> Dict[str, Any]:
        if task == "ocr":
            text = self._chat(image, ocr_type="ocr")
            return {"task": "ocr", "backend": self.name, "text": text, "html": f"<pre>{text}</pre>"}

        if task == "table":
            text = self._chat(image, ocr_type="format")
            return {
                "task": "table",
                "backend": self.name,
                "tables": [{"raw": text}],
                "note": "GOT-OCR2 format mode output",
            }

        if task == "detect":
            text = self._chat(image, ocr_type="format")
            return {
                "task": "detect",
                "backend": self.name,
                "text_lines": [{"text": text, "bbox": None, "confidence": 1.0}],
                "note": "GOT-OCR2 returns formatted text; use qwen backend for bbox JSON.",
            }

        if task == "layout":
            text = self._chat(image, ocr_type="format")
            blocks: List[Dict[str, Any]] = [
                {
                    "label": "Text",
                    "bbox": None,
                    "reading_order": 1,
                    "caption": line[:200],
                }
                for line in text.splitlines()
                if line.strip()
            ]
            return {"task": "layout", "backend": self.name, "blocks": blocks}

        raise ValueError(f"Unsupported task for gotocr2: {task}")
