from __future__ import annotations

import argparse
import json
import sys

from vlm_pipeline.pipeline import DocumentVLMPipeline
from vlm_pipeline.utils.models import (
    BACKEND_MODEL_IDS,
    download_all_models,
    model_dir,
    prune_all_models,
)


def parse_page_range(value: str) -> list[int]:
    if "-" in value:
        start, end = value.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def cmd_download(args: argparse.Namespace) -> None:
    backends = args.backend or list(BACKEND_MODEL_IDS)
    for backend in backends:
        model_id = BACKEND_MODEL_IDS[backend]
        folder = model_dir(model_id, args.models_dir)
        print(f"Downloading {model_id} -> {folder}")
    paths = download_all_models(args.models_dir, backends)
    for backend, path in paths.items():
        print(f"  done: {path}")
    print("\nModel folders:")
    for backend, model_id in BACKEND_MODEL_IDS.items():
        folder = model_dir(model_id, args.models_dir)
        print(f"  {backend:<10}  {folder}  <- {model_id}")


def cmd_prune(args: argparse.Namespace) -> None:
    counts = prune_all_models(args.models_dir)
    if not counts:
        print("No model folders to prune.")
        return
    for name, removed in counts.items():
        print(f"  {name}: removed {removed} extra files")


def cmd_run(args: argparse.Namespace) -> None:
    tasks = args.task or (None if not args.fast else ["ocr"])
    pipeline = DocumentVLMPipeline(
        backend=args.backend,
        model_id=args.model_id,
        quant=args.quant,
        fast=args.fast,
        dpi=args.dpi,
        max_image_size=args.max_image_size,
        max_tokens=args.max_tokens,
        models_dir=None if args.hf_cache else args.models_dir,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Open-source VLM document pipeline (Surya-style prompts, no Surya model)."
    )
    subparsers = parser.add_subparsers(dest="command")

    run = subparsers.add_parser("run", help="Run OCR/layout/detect on a document (default)")
    run.add_argument("input", help="PDF or image path")
    run.add_argument(
        "--backend",
        choices=list(BACKEND_MODEL_IDS),
        default="florence2",
        help="VLM backend (default: florence2)",
    )
    run.add_argument(
        "--model-id",
        default=None,
        help="Override HF model id (e.g. microsoft/Florence-2-large)",
    )
    run.add_argument(
        "--task",
        action="append",
        choices=["layout", "ocr", "table", "detect"],
        help="Task to run (repeatable). Default: ocr",
    )
    run.add_argument(
        "--page",
        type=int,
        default=0,
        help="Single PDF page index to process (default: 0 = first page only)",
    )
    run.add_argument(
        "--quant",
        choices=["auto", "none", "int4", "int8", "4bit", "8bit"],
        default="auto",
        help="Quantization: auto=int4 on CPU, 4bit on GPU",
    )
    run.add_argument(
        "--fast",
        action="store_true",
        help="Fast CPU mode: 96 DPI, 1024px image, 1536 tokens, OCR only",
    )
    run.add_argument("--dpi", type=int, default=None, help="Override render DPI")
    run.add_argument(
        "--max-image-size",
        type=int,
        default=None,
        help="Resize longest image side before model (default: 1024, fast: 768)",
    )
    run.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Max generated tokens (default: 2048, fast: 1024)",
    )
    run.add_argument(
        "--page-range",
        default=None,
        help="Optional multi-page range (e.g. 0-2). Ignored if you only use --page.",
    )
    run.add_argument("--output-dir", default="vlm_output", help="Output directory")
    run.add_argument(
        "--models-dir",
        default="models",
        help="Load model weights from models/<model_name>/ (default: models)",
    )
    run.add_argument(
        "--hf-cache",
        action="store_true",
        help="Load from Hugging Face cache instead of --models-dir",
    )
    run.add_argument("--no-images", action="store_true", help="Skip bbox overlay images")
    run.set_defaults(func=cmd_run)

    download = subparsers.add_parser("download", help="Download model weights into models/")
    download.add_argument(
        "--backend",
        action="append",
        choices=list(BACKEND_MODEL_IDS),
        help="Backend to download (repeatable). Default: all three",
    )
    download.add_argument(
        "--models-dir",
        default="models",
        help="Download into models/<model_name>/ (default: models)",
    )
    download.set_defaults(func=cmd_download)

    prune = subparsers.add_parser("prune-models", help="Remove HF clutter from model folders")
    prune.add_argument(
        "--models-dir",
        default="models",
        help="Prune models/<model_name>/ folders (default: models)",
    )
    prune.set_defaults(func=cmd_prune)

    return parser


def main() -> None:
    argv = sys.argv[1:]
    parser = build_parser()

    # Backward compatible: `python -m vlm_pipeline page.png --backend florence2`
    if argv and argv[0] not in {"run", "download", "prune-models", "-h", "--help"}:
        argv = ["run", *argv]

    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
