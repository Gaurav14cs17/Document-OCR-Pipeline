from __future__ import annotations

from typing import List

from PIL import Image

from vlm_pipeline.utils.image import pad_to_square
from vlm_pipeline.utils.text import clean_html_tags


def quad_to_bbox(quad: List[float]) -> List[int]:
    xs = quad[0::2]
    ys = quad[1::2]
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]


def pad_info(image: Image.Image) -> tuple[Image.Image, int, int, int, int]:
    """Pad to square; return (padded_image, pad_x, pad_y, orig_w, orig_h)."""
    orig_w, orig_h = image.size
    padded = pad_to_square(image)
    side = max(orig_w, orig_h)
    pad_x = (side - orig_w) // 2
    pad_y = (side - orig_h) // 2
    return padded, pad_x, pad_y, orig_w, orig_h


def unmap_bbox(bbox: List[int], pad_x: int, pad_y: int, orig_w: int, orig_h: int) -> List[int]:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(orig_w, x1 - pad_x))
    y1 = max(0, min(orig_h, y1 - pad_y))
    x2 = max(0, min(orig_w, x2 - pad_x))
    y2 = max(0, min(orig_h, y2 - pad_y))
    if x2 <= x1 or y2 <= y1:
        return []
    return [x1, y1, x2, y2]


def clean_florence_text(text: str) -> str:
    return clean_html_tags(text)
