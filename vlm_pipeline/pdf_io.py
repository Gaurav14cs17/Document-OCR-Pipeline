from __future__ import annotations

from pathlib import Path
from typing import List

from PIL import Image


def load_pages(
    input_path: str | Path,
    dpi: int = 200,
    page_range: List[int] | None = None,
) -> List[Image.Image]:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _pdf_to_images(path, dpi=dpi, page_range=page_range)
    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}:
        return [Image.open(path).convert("RGB")]
    raise ValueError(f"Unsupported input type: {suffix}")


def _pdf_to_images(
    pdf_path: Path,
    dpi: int,
    page_range: List[int] | None,
) -> List[Image.Image]:
    try:
        from pdf2image import convert_from_path
    except ImportError as exc:
        raise ImportError(
            "pdf2image is required for PDF input. Install with: pip install pdf2image"
        ) from exc

    if page_range is None:
        pages = convert_from_path(str(pdf_path), dpi=dpi)
        return [p.convert("RGB") for p in pages]

    images: List[Image.Image] = []
    for page_idx in page_range:
        page_num = page_idx + 1
        batch = convert_from_path(
            str(pdf_path),
            dpi=dpi,
            first_page=page_num,
            last_page=page_num,
        )
        if not batch:
            raise ValueError(f"Page index out of range: {page_idx}")
        images.append(batch[0].convert("RGB"))
    return images
