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
| 05 | [Production Issues](05_mobile_production_issues.ipynb) | Papers + Identify → Solve: KV, power, quant, RAM (40+ paper refs) |

---

## Pipeline Overview

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│  01 OCR     02 Quantize    03 Export      04 On-Device      05 Production      │
│  ────────   ──────────     ─────────      ──────────        ───────────        │
│  Florence-2 GPTQ/AWQ/     Pack int4      mmap load         KV cache OOM        │
│  detect+OCR SmoothQuant/   ONNX graph     generate loop     power / thermal      │
│  layout     SpinQuant/     mobile binary  OTA + native      quant precision      │
│             ConvRot        artifacts      templates         RAM breakdown        │
└──────────────────────────────────────────────────────────────────────────────────┘
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

**Layer coverage:** Notebook 02 surveys **all** module types and quantizes **Linear, Conv1d/2d/3d, and Embedding** in Stages 8–12. Flow: Stage 7b (`LayerProfile`) → Stage 8–9 (inventory + sensitivity) → Stage 11 (assign bits) → **Stage 12** (`apply_quant_plan` — load each weight, quantize, swap). `nn.Linear` gets all five methods; Conv uses per-channel RTN/AWQ; Embedding uses per-row RTN/AWQ.

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
- **Production diagnostics** — notebook 05 scorecard for KV cache, power, quant loss, and RAM

---

## Production Issues (Notebook 05)

**Structure: Papers → Identify → Solve → Verify (step-by-step)**

| Part | Stages | What you do |
|------|--------|-------------|
| **Literature** | Each stage opens with key papers + story arc | Read why the problem exists |
| **Part A — Identify** | Stages 2–5 | 6 detection methods per problem + code |
| **Part B — Solve** | Stages 6–9 | Paper-backed fixes ranked P0→P3 + code |
| **Part C — Verify** | Stage 10 | Before/after scorecard + bibliography |

### Key papers by topic

| Topic | Papers cited in notebook 05 |
|-------|----------------------------|
| **KV cache** | Vaswani 2017, Shazeer 2019 (MQA), Ainslie 2023 (GQA), Dao 2022/23 (FlashAttention), Kwon 2023 (PagedAttention/vLLM), H2O, Scissorhands, StreamingLLM, SnapKV, KIVI, KVQuant, LM-Infinite |
| **Power** | Yu 2022 (Orca continuous batching), Leviathan/Chen 2023 (speculative decoding), Medusa, Splitwise, PowerInfer, Apple LLM-in-a-Flash, MobileLLM |
| **Quant loss** | Dettmers 2022 (LLM.int8), GPTQ, AWQ, SmoothQuant, SpinQuant, SqueezeLLM, QuaRot, OmniQuant, QServe |
| **RAM** | Pope 2023, Sheng 2023 (FlexGen), ZeRO-Inference, ZeroQuant, Apple flash streaming, llama.cpp |

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
