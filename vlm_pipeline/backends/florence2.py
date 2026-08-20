from __future__ import annotations
from typing import Any, Dict, List
import torch
from PIL import Image
from vlm_pipeline.backends.base import VLMBackend
from vlm_pipeline.image_utils import pad_to_square


def _quad_to_bbox(quad: List[float]) -> List[int]:
    xs = quad[0::2]
    ys = quad[1::2]
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]


def _pad_info(image: Image.Image) -> tuple[Image.Image, int, int, int, int]:
    """Pad to square and return (padded_image, pad_x, pad_y, orig_w, orig_h)."""
    orig_w, orig_h = image.size
    padded = pad_to_square(image)
    side = max(orig_w, orig_h)
    pad_x = (side - orig_w) // 2
    pad_y = (side - orig_h) // 2
    return padded, pad_x, pad_y, orig_w, orig_h


def _unmap_bbox(bbox: List[int], pad_x: int, pad_y: int, orig_w: int, orig_h: int) -> List[int]:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(orig_w, x1 - pad_x))
    y1 = max(0, min(orig_h, y1 - pad_y))
    x2 = max(0, min(orig_w, x2 - pad_x))
    y2 = max(0, min(orig_h, y2 - pad_y))
    if x2 <= x1 or y2 <= y1:
        return []
    return [x1, y1, x2, y2]


def _clean_text(text: str) -> str:
    import re

    text = re.sub(r"</?[a-zA-Z_][^>]*>", "", text)
    return text.strip()


class Florence2Backend(VLMBackend):
    """Lightweight open VLM (230M/770M). Best default for CPU."""

    name = "florence2"

    def __init__(self, model_id: str = "microsoft/Florence-2-base-ft", device: str | None = None, quant: str = "auto"):
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.model_id = model_id
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        dtype = torch.float16 if device == "cuda" else torch.float32

        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=dtype,
            attn_implementation="eager",
        ).to(device)
        self.model.eval()

    def _generate(self, image: Image.Image, task_prompt: str, max_new_tokens: int = 2048) -> tuple[str, int, int, int, int]:
        padded, pad_x, pad_y, orig_w, orig_h = _pad_info(image)
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
    ) -> Dict[str, Any]:
        side = max(orig_w, orig_h)
        return self.processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(side, side),
        ), pad_x, pad_y, orig_w, orig_h

    def run_task(
        self,
        image: Image.Image,
        task: str,
        max_new_tokens: int | None = None,
    ) -> Dict[str, Any]:
        if task == "detect":
            return self._detect(image)
        if task == "ocr":
            return self._ocr(image)
        if task == "layout":
            return self._layout(image)
        if task == "table":
            return self._table(image)
        raise ValueError(f"Unsupported task for florence2: {task}")

    def _detect(self, image: Image.Image) -> Dict[str, Any]:
        task_prompt = "<OCR_WITH_REGION>"
        generated, pad_x, pad_y, orig_w, orig_h = self._generate(image, task_prompt)
        parsed, pad_x, pad_y, orig_w, orig_h = self._parse_florence(
            task_prompt, generated, pad_x, pad_y, orig_w, orig_h
        )
        region = parsed.get(task_prompt, {})
        lines = []
        for quad, label in zip(region.get("quad_boxes", []), region.get("labels", [])):
            bbox = _unmap_bbox(_quad_to_bbox(quad), pad_x, pad_y, orig_w, orig_h)
            if not bbox:
                continue
            lines.append(
                {
                    "text": _clean_text(label),
                    "bbox": bbox,
                    "confidence": 1.0,
                }
            )
        return {"task": "detect", "backend": self.name, "text_lines": lines}

    def _ocr(self, image: Image.Image) -> Dict[str, Any]:
        task_prompt = "<OCR>"
        generated, pad_x, pad_y, orig_w, orig_h = self._generate(image, task_prompt)
        parsed, _, _, _, _ = self._parse_florence(task_prompt, generated, pad_x, pad_y, orig_w, orig_h)
        text = parsed.get(task_prompt, "")
        if not isinstance(text, str):
            text = str(text)
        text = _clean_text(text)
        return {"task": "ocr", "backend": self.name, "html": f"<p>{text}</p>", "text": text}

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
            mapped = _unmap_bbox([int(v) for v in bbox], pad_x, pad_y, orig_w, orig_h)
            if not mapped:
                continue
            blocks.append(
                {
                    "label": "Text",
                    "bbox": mapped,
                    "reading_order": i,
                    "caption": _clean_text(str(label)),
                }
            )
        return {"task": "layout", "backend": self.name, "blocks": blocks}

    def _table(self, image: Image.Image) -> Dict[str, Any]:
        # Florence table extraction is limited; return OCR text and let caller post-process.
        ocr = self._ocr(image)
        return {
            "task": "table",
            "backend": self.name,
            "tables": [],
            "note": "Florence-2 has limited table structure support. Use --backend qwen for table JSON.",
            "fallback_text": ocr.get("text", ""),
        }
