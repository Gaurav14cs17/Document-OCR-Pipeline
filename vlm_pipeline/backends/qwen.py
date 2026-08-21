from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict

import torch
from PIL import Image

from vlm_pipeline.backends.base import VLMBackend
from vlm_pipeline.utils.prompts import TASK_PROMPTS
from vlm_pipeline.utils.quant import build_load_kwargs
from vlm_pipeline.utils.text import clean_model_response, extract_json


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

        self.model_id = model_id
        self.device = self.resolve_device(device)

        quant_cfg = build_load_kwargs(quant, self.device)
        self.quant_mode = quant_cfg["quant_mode"]
        load_kwargs = quant_cfg["load_kwargs"]

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
        if self.device != "cuda" and load_kwargs.get("device_map") is None:
            self.model = self.model.to(self.device)
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

            input_len = inputs["input_ids"].shape[1]
            new_tokens = generated[:, input_len:]
            return self.processor.batch_decode(
                new_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def _run_prompt_task(self, image: Image.Image, task: str, max_new_tokens: int | None) -> Dict[str, Any]:
        self.validate_task(task)
        cleaned = clean_model_response(
            self._ask(image, TASK_PROMPTS[task], max_new_tokens=max_new_tokens or 2048)
        )
        if task == "ocr":
            return self.task_result("ocr", html=cleaned, text=cleaned)
        try:
            parsed = extract_json(cleaned)
        except json.JSONDecodeError:
            parsed = {"raw": cleaned}
        result = self.task_result(task)
        result.update(parsed if isinstance(parsed, dict) else {"raw": parsed})
        return result

    def run_task(self, image: Image.Image, task: str, max_new_tokens: int | None = None) -> Dict[str, Any]:
        return self._run_prompt_task(image, task, max_new_tokens)