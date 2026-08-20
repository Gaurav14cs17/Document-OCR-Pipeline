from __future__ import annotations

from pathlib import Path

DEFAULT_MODELS_DIR = "models"

BACKEND_MODEL_IDS = {
    "florence2": "microsoft/Florence-2-base-ft",
    "qwen": "Qwen/Qwen2.5-VL-3B-Instruct",
    "gotocr2": "stepfun-ai/GOT-OCR2_0",
}

DOWNLOAD_IGNORE_PATTERNS = [
    "*.md",
    "LICENSE*",
    "CODE_OF_CONDUCT*",
    "SECURITY*",
    "SUPPORT*",
    ".git*",
    "assets/**",
    "*.jpg",
    "*.png",
]

PRUNE_NAMES = frozenset({
    "README.md",
    "LICENSE",
    "CODE_OF_CONDUCT.md",
    "SECURITY.md",
    "SUPPORT.md",
    ".gitattributes",
    ".gitignore",
})


def model_folder_name(model_id: str) -> str:
    """HF repo id -> local folder (e.g. microsoft/Florence-2-base-ft -> Florence-2-base-ft)."""
    return model_id.rsplit("/", 1)[-1]


def model_dir(model_id: str, models_dir: str | Path = DEFAULT_MODELS_DIR) -> Path:
    return Path(models_dir) / model_folder_name(model_id)


def is_model_present(path: Path) -> bool:
    if not path.is_dir():
        return False
    markers = ("config.json", "preprocessor_config.json", "model.safetensors", "pytorch_model.bin")
    return any((path / name).exists() for name in markers)


def prune_model_dir(model_path: str | Path) -> int:
    """Remove HF download clutter; keep only files needed to load the model."""
    root = Path(model_path)
    if not root.is_dir():
        return 0

    removed = 0
    for path in sorted(root.rglob("*"), reverse=True):
        if not path.is_file():
            continue
        name = path.name
        if name.endswith((".lock", ".metadata")) or name in PRUNE_NAMES:
            path.unlink(missing_ok=True)
            removed += 1

    assets_dir = root / "assets"
    if assets_dir.is_dir():
        for path in sorted(assets_dir.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink(missing_ok=True)
                removed += 1
        for path in sorted(assets_dir.rglob("*"), reverse=True):
            if path.is_dir():
                path.rmdir()
        assets_dir.rmdir()
        removed += 1

    safetensors = list(root.glob("*.safetensors")) + list(root.glob("model-*-of-*.safetensors"))
    if safetensors:
        bin_path = root / "pytorch_model.bin"
        if bin_path.is_file():
            bin_path.unlink(missing_ok=True)
            removed += 1

    cache_dir = root / "__pycache__"
    if cache_dir.is_dir():
        for path in sorted(cache_dir.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink(missing_ok=True)
                removed += 1
        cache_dir.rmdir()
        removed += 1

    return removed


def ensure_local_model(model_id: str, models_dir: str | Path = DEFAULT_MODELS_DIR) -> Path:
    """Download weights into models/<model_name>/ if not already present."""
    local_path = model_dir(model_id, models_dir)
    if is_model_present(local_path):
        prune_model_dir(local_path)
        return local_path

    local_path.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to download models. Install with: pip install huggingface_hub"
        ) from exc

    snapshot_download(
        repo_id=model_id,
        local_dir=str(local_path),
        ignore_patterns=DOWNLOAD_IGNORE_PATTERNS,
    )
    prune_model_dir(local_path)
    return local_path


def resolve_model_source(
    model_id: str,
    models_dir: str | Path | None = DEFAULT_MODELS_DIR,
) -> str:
    if models_dir is None:
        return model_id
    return str(ensure_local_model(model_id, models_dir))


def prune_all_models(models_dir: str | Path = DEFAULT_MODELS_DIR) -> dict[str, int]:
    """Prune every downloaded model folder under models_dir."""
    root = Path(models_dir)
    counts: dict[str, int] = {}
    for model_id in BACKEND_MODEL_IDS.values():
        path = model_dir(model_id, root)
        if path.is_dir():
            counts[path.name] = prune_model_dir(path)
    pycache = root / "__pycache__"
    if pycache.is_dir():
        for path in sorted(pycache.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink(missing_ok=True)
        pycache.rmdir()
    return counts


def download_all_models(
    models_dir: str | Path = DEFAULT_MODELS_DIR,
    backends: list[str] | None = None,
) -> dict[str, str]:
    """Download all (or selected) backend models into models_dir."""
    names = backends or list(BACKEND_MODEL_IDS)
    paths: dict[str, str] = {}
    for backend in names:
        if backend not in BACKEND_MODEL_IDS:
            raise ValueError(f"Unknown backend '{backend}'. Choose from: {list(BACKEND_MODEL_IDS)}")
        model_id = BACKEND_MODEL_IDS[backend]
        local_path = ensure_local_model(model_id, models_dir)
        paths[backend] = str(local_path)
    return paths
