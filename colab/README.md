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
| 02a | [GPTQ Visual Walkthrough](02_ocr_pipeline_quant_1A.ipynb) | **Start here for GPTQ** — 3×3 toy matrices, heatmaps, hooks explained step-by-step (~10 min) |
| 02b | [GPTQ on Florence-2](02_ocr_pipeline_quant_A.ipynb) | Production GPTQ on `nn.Linear` only — scan → hook → rank → mixed-precision plan → swap |
| 02c | [Five Quant Methods](02_ocr_pipeline_quant_B.ipynb) | GPTQ · AWQ · SmoothQuant · SpinQuant · ConvRot — Phases A–E with mixed-precision planning |
| 02d | [Activation Calibration](02_ocr_pipeline_quant_C.ipynb) | 6 observers (MinMax, EMA, percentile, KL, MSE) — hooks + scales, no weight rounding |
| 03 | [Mobile Export](03_ocr_pipeline_mobile.ipynb) | Packed int4 weights, ONNX export, mobile-ready artifacts |
| 04 | [Mobile Complete](04_ocr_pipeline_mobile_complete.ipynb) | OTA delivery, mmap loading, autoregressive generate loop, native templates |
| 05 | [Production Issues](05_mobile_production_issues.ipynb) | Papers + Identify → Solve: KV, power, quant, RAM (40+ paper refs) |

### Recommended order

```
01 OCR  →  02a (optional warm-up)  →  02b OR 02c  →  02d (optional)  →  03  →  04  →  05
```

- **New to GPTQ?** Run **02a** first — tiny 3×3 network, one class per cell, visual proof of hook capture.
- **Want one method on Florence-2?** Run **02b** (GPTQ + mixed precision, cleaner scope).
- **Want all five methods compared?** Run **02c** (full Phases A–E, Stage 16 OCR scorecard).
- **Care about activation scales?** Run **02d** after 02b or 02c.

---

## Pipeline Overview

```
┌────────────────────────────────────────────────────────────────────────────────────────────┐
│  01 OCR     02 Quant (a→b/c→d)   03 Export      04 On-Device      05 Production            │
│  ────────   ─────────────────    ─────────      ──────────        ───────────            │
│  Florence-2 GPTQ/AWQ/            Pack int4      mmap load         KV cache OOM             │
│  detect+OCR SmoothQuant/           ONNX graph     generate loop     power / thermal          │
│  layout     SpinQuant/             mobile binary  OTA + native      quant precision          │
│             ConvRot + observers    artifacts      templates         RAM breakdown            │
└────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

1. Open any notebook in [Google Colab](https://colab.research.google.com)
2. Select **GPU runtime** (T4 is sufficient)
3. Run all cells top-to-bottom

Each notebook is self-contained (installs its own dependencies) but follows the series order for the full pipeline experience.

If pip updates `transformers` or `numpy`, Colab may ask you to **restart runtime** — then click **Run all**.

---

## Key Features

- **No external quant libraries** — every algorithm built from first principles
- **Readable hooks** — named recorder classes + `register_layer_hooks` / `remove_hooks` (no nested closures)
- **Teaching notebook (02a)** — 3×3 matrices, heatmaps, one concept per cell
- **Mixed-precision planning** — sensitivity-driven int4/int8/fp16 per layer
- **5-method comparison** — Stage 16 in **02c** runs all quantizers on the same OCR task
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

Notebook **01** also supports Qwen-VL backends; Florence-2 is the default for the quant and mobile series.

---

## Requirements

- Python 3.10+
- PyTorch 2.x
- `transformers==4.49.0` (Florence-2 notebooks 02–05; notebook 01 pins 4.49 for Florence-2 or ≥4.51 for Qwen-VL)
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
