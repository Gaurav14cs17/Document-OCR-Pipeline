from __future__ import annotations

from typing import Any, Dict

import torch
from PIL import Image

from vlm_pipeline.backends.base import VLMBackend
from vlm_pipeline.utils.florence import (
    clean_florence_text,
    pad_info,
    quad_to_bbox,
    unmap_bbox,
)


class Florence2Backend(VLMBackend):
    """Lightweight open VLM (230M/770M). Best default for CPU."""

    name = "florence2"

    def __init__(
        self,
        model_id: str = "microsoft/Florence-2-base-ft",
        device: str | None = None,
        quant: str = "auto",
    ):
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.model_id = model_id
        self.device = self.resolve_device(device)
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=dtype,
            attn_implementation="eager",
        ).to(self.device)
        self.model.eval()

    def _generate(self, image: Image.Image, task_prompt: str, max_new_tokens: int = 2048) -> tuple[str, int, int, int, int]:
        padded, pad_x, pad_y, orig_w, orig_h = pad_info(image)
        inputs = self.processor(
            text=task_prompt,
            images=padded,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            generated = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=max_new_tokens,
                num_beams=1,
                use_cache=False,
            )

        text = self.processor.batch_decode(generated, skip_special_tokens=False)[0]
        return text, pad_x, pad_y, orig_w, orig_h

    def _parse_florence(
        self,
        task_prompt: str,
        generated_text: str,
        pad_x: int,
        pad_y: int,
        orig_w: int,
        orig_h: int,
    ) -> tuple[Dict[str, Any], int, int, int, int]:
        side = max(orig_w, orig_h)
        parsed = self.processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(side, side),
        )
        return parsed, pad_x, pad_y, orig_w, orig_h

    def _handlers(self) -> dict[str, Any]:
        return {
            "detect": self._detect,
            "ocr": self._ocr,
            "layout": self._layout,
            "table": self._table,
        }

    def run_task(self, image: Image.Image, task: str, max_new_tokens: int | None = None) -> Dict[str, Any]:
        return self._dispatch(image, task, self._handlers())

    def _detect(self, image: Image.Image) -> Dict[str, Any]:
        task_prompt = "<OCR_WITH_REGION>"
        generated, pad_x, pad_y, orig_w, orig_h = self._generate(image, task_prompt)
        parsed, pad_x, pad_y, orig_w, orig_h = self._parse_florence(
            task_prompt, generated, pad_x, pad_y, orig_w, orig_h
        )
        region = parsed.get(task_prompt, {})
        lines = []
        for quad, label in zip(region.get("quad_boxes", []), region.get("labels", [])):
            bbox = unmap_bbox(quad_to_bbox(quad), pad_x, pad_y, orig_w, orig_h)
            if not bbox:
                continue
            lines.append({"text": clean_florence_text(label), "bbox": bbox, "confidence": 1.0})
        return self.task_result("detect", text_lines=lines)

    def _ocr(self, image: Image.Image) -> Dict[str, Any]:
        task_prompt = "<OCR>"
        generated, pad_x, pad_y, orig_w, orig_h = self._generate(image, task_prompt)
        parsed, _, _, _, _ = self._parse_florence(task_prompt, generated, pad_x, pad_y, orig_w, orig_h)
        text = parsed.get(task_prompt, "")
        if not isinstance(text, str):
            text = str(text)
        text = clean_florence_text(text)
        return self.task_result("ocr", html=f"<p>{text}</p>", text=text)

    def _layout(self, image: Image.Image) -> Dict[str, Any]:
        task_prompt = "<DENSE_REGION_CAPTION>"
        generated, pad_x, pad_y, orig_w, orig_h = self._generate(image, task_prompt, max_new_tokens=1024)
        parsed, pad_x, pad_y, orig_w, orig_h = self._parse_florence(
            task_prompt, generated, pad_x, pad_y, orig_w, orig_h
        )
        labels = parsed.get(task_prompt, {}).get("labels", [])
        bboxes = parsed.get(task_prompt, {}).get("bboxes", [])
        blocks = []
        for i, (bbox, label) in enumerate(zip(bboxes, labels), start=1):
            mapped = unmap_bbox([int(v) for v in bbox], pad_x, pad_y, orig_w, orig_h)
            if not mapped:
                continue
            blocks.append(
                {
                    "label": "Text",
                    "bbox": mapped,
                    "reading_order": i,
                    "caption": clean_florence_text(str(label)),
                }
            )
        return self.task_result("layout", blocks=blocks)

    def _table(self, image: Image.Image) -> Dict[str, Any]:
        ocr = self._ocr(image)
        return self.task_result(
            "table",
            tables=[],
            note="Florence-2 has limited table structure support. Use --backend qwen for table JSON.",
            fallback_text=ocr.get("text", ""),
        )
