from __future__ import annotations

import torch


def resolve_quant_mode(quant: str, device: str) -> str:
    if quant == "auto":
        return "4bit" if device == "cuda" else "int4"
    return quant


def build_load_kwargs(quant: str, device: str) -> dict:
    """Build from_pretrained kwargs for quantized loading."""
    mode = resolve_quant_mode(quant, device)
    kwargs: dict = {"device_map": "auto" if device == "cuda" else None}

    if mode == "none":
        kwargs["torch_dtype"] = torch.bfloat16 if device == "cuda" else torch.float32
        return {"quant_mode": mode, "load_kwargs": kwargs}

    if device == "cuda" and mode in {"4bit", "8bit"}:
        from transformers import BitsAndBytesConfig

        if mode == "4bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        else:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        return {"quant_mode": mode, "load_kwargs": kwargs}

    if device == "cpu" and mode in {"int4", "int8"}:
        from transformers import QuantoConfig

        kwargs["quantization_config"] = QuantoConfig(weights=mode)
        kwargs["torch_dtype"] = torch.float32
        kwargs["low_cpu_mem_usage"] = True
        return {"quant_mode": mode, "load_kwargs": kwargs}

    raise ValueError(
        f"Unsupported quant mode '{mode}' for device '{device}'. "
        "Use int4/int8 on CPU, 4bit/8bit on GPU, or none."
    )
