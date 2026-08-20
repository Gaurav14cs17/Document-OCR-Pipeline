from __future__ import annotations

import argparse
import json

from vlm_pipeline.pipeline import DocumentVLMPipeline


def parse_page_range(value: str) -> list[int]:
    if "-" in value:
        start, end = value.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Open-source VLM document pipeline (Surya-style prompts, no Surya model)."
    )
    parser.add_argument("input", help="PDF or image path")
    parser.add_argument(
        "--backend",
        choices=["gotocr2", "florence2", "qwen"],
        default="qwen",
        help="VLM backend (qwen=Surya-like prompts, gotocr2=GPU OCR VLM)",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Override HF model id (e.g. microsoft/Florence-2-large)",
    )
    parser.add_argument(
        "--task",
        action="append",
        choices=["layout", "ocr", "table", "detect"],
        help="Task to run (repeatable). Default: ocr (single page)",
    )
    parser.add_argument(
        "--page",
        type=int,
        default=0,
        help="Single PDF page index to process (default: 0 = first page only)",
    )
    parser.add_argument(
        "--quant",
        choices=["auto", "none", "int4", "int8", "4bit", "8bit"],
        default="auto",
        help="Quantization: auto=int4 on CPU, 4bit on GPU",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast CPU mode: 96 DPI, 1024px image, 1536 tokens, OCR only",
    )
    parser.add_argument("--dpi", type=int, default=None, help="Override render DPI")
    parser.add_argument(
        "--max-image-size",
        type=int,
        default=None,
        help="Resize longest image side before model (default: 1024, fast: 768)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Max generated tokens (default: 2048, fast: 1024)",
    )
    parser.add_argument(
        "--page-range",
        default=None,
        help="Optional multi-page range (e.g. 0-2). Ignored if you only use --page.",
    )
    parser.add_argument("--output-dir", default="vlm_output", help="Output directory")
    parser.add_argument("--no-images", action="store_true", help="Skip bbox overlay images")
    args = parser.parse_args()

    tasks = args.task or (None if not args.fast else ["ocr"])
    pipeline = DocumentVLMPipeline(
        backend=args.backend,
        model_id=args.model_id,
        quant=args.quant,
        fast=args.fast,
        dpi=args.dpi,
        max_image_size=args.max_image_size,
        max_tokens=args.max_tokens,
    )
    page_range = (
        parse_page_range(args.page_range) if args.page_range is not None else [args.page]
    )
    result = pipeline.run(
        input_path=args.input,
        tasks=tasks,
        page_range=page_range,
        output_dir=args.output_dir,
        save_images=not args.no_images,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
