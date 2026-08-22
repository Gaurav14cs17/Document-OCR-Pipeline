# Document OCR Pipeline — Colab Notebook Series

End-to-end pipeline from **pixels to on-device OCR**, implemented from scratch in PyTorch.

```
Image → VLM OCR → Quantize → Export → Mobile Inference
```

---

## Notebooks

| # | Notebook | Focus |
|---|----------|-------|
| 01 | [Document OCR Pipeline](01_document_ocr_pipeline.ipynb) | Florence-2 OCR walkthrough — detection, recognition, layout, tables |
| 02 | [Quantization](02_ocr_pipeline_quant.ipynb) | GPTQ · AWQ · SmoothQuant · SpinQuant · ConvRot (all from scratch) |
| 03 | [Mobile Export](03_ocr_pipeline_mobile.ipynb) | Packed int4 weights, ONNX export, mobile-ready artifacts |
| 04 | [Mobile Complete](04_ocr_pipeline_mobile_complete.ipynb) | OTA delivery, mmap loading, autoregressive generate loop, native templates |

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│  01 OCR          02 Quantize         03 Export         04 On-Device     │
│  ────────        ──────────          ─────────         ──────────       │
│  Florence-2      GPTQ / AWQ /        Pack int4         mmap load        │
│  detect+OCR      SmoothQuant /       ONNX graph        generate loop    │
│  layout+table    SpinQuant /         mobile binary     OTA updates      │
│                  ConvRot (W4A4)                        JNI bridge       │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Quantization Methods (Notebook 02)

| Method | Core Idea | Complexity |
|--------|-----------|------------|
| **GPTQ** | Hessian-aware column-wise quantization | Per-column Cholesky update |
| **AWQ** | Activation-aware per-channel scale search | Grid search over scales |
| **SmoothQuant** | Migrate outliers from activations to weights | Per-channel diagonal $s_j$ |
| **SpinQuant** | Learned Givens rotations before quantize | $O(K \cdot n_\text{rotations})$ |
| **ConvRot** | Group-wise Regular Hadamard Transform (RHT) | $O(K)$ — plug-and-play W4A4 |

All implemented **from scratch** in PyTorch — no quantization libraries required.

---

## Quick Start

1. Open any notebook in [Google Colab](https://colab.research.google.com)
2. Select **GPU runtime** (T4 is sufficient)
3. Run all cells top-to-bottom

Each notebook is self-contained (installs its own dependencies) but follows the series order for the full pipeline experience.

---

## Key Features

- **Blog-style prose** with LaTeX math and formal proofs throughout
- **No external quant libraries** — every algorithm built from first principles
- **Mixed-precision planning** — sensitivity-driven int4/int8/fp16 per layer
- **5-method comparison** — Stage 14 runs all quantizers on the same OCR task (notebook 02)
- **Mobile-ready** — packed binaries, ONNX, mmap, autoregressive decode

---

## Model

**microsoft/Florence-2-base-ft** — a compact vision-language model for document understanding tasks (OCR, detection, layout analysis, table recognition).

---

## Requirements

- Python 3.10+
- PyTorch 2.x
- `transformers==4.49.0`
- Google Colab GPU runtime (or local CUDA)

All dependencies are installed automatically in each notebook's first cell.

---

## Papers

| Method | Authors | Venue | Link |
|--------|---------|-------|------|
| **GPTQ** | Frantar et al. | ICLR 2023 | [arxiv.org/abs/2210.17323](https://arxiv.org/abs/2210.17323) |
| **AWQ** | Lin et al. | MLSys 2024 | [arxiv.org/abs/2306.00978](https://arxiv.org/abs/2306.00978) |
| **SmoothQuant** | Xiao et al. | ICML 2023 | [arxiv.org/abs/2211.10438](https://arxiv.org/abs/2211.10438) |
| **SpinQuant** | Liu et al. | ICLR 2025 | [arxiv.org/abs/2405.16406](https://arxiv.org/abs/2405.16406) |
| **ConvRot** | Huang et al. | arXiv 2024 | [arxiv.org/abs/2512.03673](https://arxiv.org/abs/2512.03673) |
| **Florence-2** | Xiao et al. | CVPR 2024 | [arxiv.org/abs/2311.06242](https://arxiv.org/abs/2311.06242) |
