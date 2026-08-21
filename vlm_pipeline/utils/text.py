from __future__ import annotations

import json
import re
from typing import Any


def clean_html_tags(text: str) -> str:
    return re.sub(r"</?[a-zA-Z_][^>]*>", "", text).strip()


def clean_model_response(text: str) -> str:
    text = text.strip()
    if "assistant" in text:
        text = text.split("assistant", 1)[-1].strip()
    fence = re.search(r"```(?:html|json)?\s*([\s\S]*?)\s*```", text)
    if fence:
        return fence.group(1).strip()
    return text


def extract_json(text: str) -> Any:
    text = clean_model_response(text)
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            return json.loads(match.group(0))
        raise
