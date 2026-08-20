from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image, ImageDraw, ImageFont


def _clean_label(text: str) -> str:
    text = re.sub(r"</?[a-zA-Z_][^>]*>", "", text)
    return text.strip()


def _draw_boxes(
    image: Image.Image,
    boxes: List[Dict[str, Any]],
    bbox_key: str = "bbox",
    label_key: str = "text",
    color: str = "red",
) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.load_default(size=11)
    except TypeError:
        font = ImageFont.load_default()

    for item in boxes:
        bbox = item.get(bbox_key)
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox]
        if x2 <= x1 or y2 <= y1:
            continue

        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = _clean_label(str(item.get(label_key, "")))[:60]
        if not label:
            continue

        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        box_w = x2 - x1
        label_x = x1 if text_w <= box_w else max(0, x1)
        label_y = min(y2 - text_h - 2, image.height - text_h - 2)
        label_y = max(y1, label_y)

        draw.rectangle(
            [label_x, label_y, label_x + min(text_w + 4, box_w), label_y + text_h + 2],
            fill="white",
            outline=color,
        )
        draw.text((label_x + 2, label_y + 1), label, fill=color, font=font)
    return out


def _side_by_side(
    left: Image.Image,
    right: Image.Image,
    left_label: str = "Input",
    right_label: str = "Output",
    gap: int = 16,
    header_h: int = 32,
) -> Image.Image:
    """Stack input (left) and annotated output (right) with labels."""
    target_h = max(left.height, right.height)

    def _fit_height(img: Image.Image) -> Image.Image:
        if img.height == target_h:
            return img
        new_w = max(1, int(img.width * target_h / img.height))
        return img.resize((new_w, target_h), Image.Resampling.LANCZOS)

    left = _fit_height(left)
    right = _fit_height(right)

    total_w = left.width + gap + right.width
    canvas = Image.new("RGB", (total_w, header_h + target_h), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default(size=12)
    except TypeError:
        font = ImageFont.load_default()

    draw.text((12, 8), left_label, fill="#333333", font=font)
    draw.text((left.width + gap + 12, 8), right_label, fill="#333333", font=font)
    draw.line([(0, header_h - 1), (total_w, header_h - 1)], fill="#cccccc", width=1)
    canvas.paste(left, (0, header_h))
    canvas.paste(right, (left.width + gap, header_h))
    return canvas


def save_page_comparison(
    image: Image.Image,
    page_tasks: Dict[str, Any],
    output_dir: Path,
    page_index: int,
) -> str | None:
    """Save input | output side-by-side PNG for the page."""
    output_dir.mkdir(parents=True, exist_ok=True)

    overlay = None
    if "detect" in page_tasks and page_tasks["detect"].get("text_lines"):
        overlay = _draw_boxes(image, page_tasks["detect"]["text_lines"], label_key="text")
    elif "layout" in page_tasks and page_tasks["layout"].get("blocks"):
        overlay = _draw_boxes(image, page_tasks["layout"]["blocks"], label_key="label", color="blue")

    if overlay is None:
        return None

    comparison = _side_by_side(image, overlay)
    path = output_dir / f"page_{page_index}_comparison.png"
    comparison.save(path)
    return str(path)


def save_overlays(
    image: Image.Image,
    page_result: Dict[str, Any],
    output_dir: Path,
    page_index: int,
) -> List[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: List[str] = []

    if "text_lines" in page_result:
        vis = _draw_boxes(image, page_result["text_lines"], label_key="text")
        path = output_dir / f"page_{page_index}_detect_bbox.png"
        vis.save(path)
        saved.append(str(path))

    if "blocks" in page_result:
        vis = _draw_boxes(image, page_result["blocks"], label_key="label", color="blue")
        path = output_dir / f"page_{page_index}_layout_bbox.png"
        vis.save(path)
        saved.append(str(path))

    return saved
