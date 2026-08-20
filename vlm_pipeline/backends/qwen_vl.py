from __future__ import annotations
import json
import re
import tempfile
from pathlib import Path
from typing import Any, Dict
import torch
from PIL import Image
from vlm_pipeline.backends.base import VLMBackend
from vlm_pipeline.prompts import TASK_PROMPTS


def _extract_json(text: str) -> Any:
    text = _clean_response(text)
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fence:
        text = fence.group(1).strip()
    return json.loads(text)


def _clean_response(text: str) -> str:
    text = text.strip()
    if "assistant" in text:
        text = text.split("assistant", 1)[-1].strip()
    fence = re.search(r"```(?:html|json)?\s*([\s\S]*?)\s*```", text)
    if fence:
        return fence.group(1).strip()
    return text


class QwenVLBackend(VLMBackend):
    """Qwen2.5-VL open VLM. Best quality; GPU recommended."""

    name = "qwen"

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: str | None = None,
        quant: str = "auto",
    ):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        from vlm_pipeline.quant import build_load_kwargs

        self.model_id = model_id
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        quant_cfg = build_load_kwargs(quant, device)
        self.quant_mode = quant_cfg["quant_mode"]
        load_kwargs = quant_cfg["load_kwargs"]

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            **load_kwargs,
        )
        if device != "cuda" and load_kwargs.get("device_map") is None:
            self.model = self.model.to(device)
        self.processor = AutoProcessor.from_pretrained(model_id)

    def _ask(self, image: Image.Image, prompt: str, max_new_tokens: int = 4096) -> str:
        from qwen_vl_utils import process_vision_info

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            temp_path = tmp.name
        image.save(temp_path)

        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": temp_path},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.model.device)

            with torch.no_grad():
                generated = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

            return self.processor.batch_decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def run_task(
        self,
        image: Image.Image,
        task: str,
        max_new_tokens: int | None = None,
    ) -> Dict[str, Any]:
        if task not in TASK_PROMPTS:
            raise ValueError(f"Unsupported task: {task}")

        prompt = TASK_PROMPTS[task]
        tokens = max_new_tokens or 2048
        raw = self._ask(image, prompt, max_new_tokens=tokens)
        cleaned = _clean_response(raw)

        if task == "ocr":
            return {"task": "ocr", "backend": self.name, "html": cleaned, "text": cleaned}

        try:
            parsed = _extract_json(cleaned)
        except json.JSONDecodeError:
            parsed = {"raw": cleaned}

        result: Dict[str, Any] = {"task": task, "backend": self.name}
        result.update(parsed if isinstance(parsed, dict) else {"raw": parsed})
        return result
