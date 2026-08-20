from __future__ import annotations

from PIL import Image


def pad_to_square(image: Image.Image, fill: str = "white") -> Image.Image:
    """Pad image to a square canvas (required by some vision encoders)."""
    w, h = image.size
    if w == h:
        return image
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), fill)
    canvas.paste(image, ((side - w) // 2, (side - h) // 2))
    return canvas


def resize_long_edge(image: Image.Image, max_size: int) -> Image.Image:
    """Downscale so the longest side is at most max_size pixels."""
    w, h = image.size
    long_edge = max(w, h)
    if long_edge <= max_size:
        return image
    scale = max_size / long_edge
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS)
