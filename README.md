# Titan-LM

<p align="center">
  <b>A Bare-Metal Transformer Engine Written from Scratch in Pure C++20 for Apple Silicon M4</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/C%2B%2B-20-00599C?logo=c%2B%2B" alt="C++20">
  <img src="https://img.shields.io/badge/Hardware-Apple%20Silicon%20M4%20(ARM64)-black?logo=apple" alt="Apple Silicon M4">
  <img src="https://img.shields.io/badge/Acceleration-Apple%20Accelerate%20(vDSP%2Fcblas)-green" alt="Apple Accelerate">
  <img src="https://img.shields.io/badge/Dependencies-Zero%20External%20Libs-orange" alt="Zero Dependencies">
  <img src="https://img.shields.io/badge/License-MIT-blue" alt="License">
</p>

---

## Overview

**Titan-LM** is an end-to-end, high-performance Transformer Large Language Model implemented entirely from first principles in pure C++20. It contains **zero external machine learning dependencies**—no PyTorch, libtorch, GGML, ONNX, or Python runtime.

Engineered specifically for the **Apple Silicon M4** architecture, Titan-LM leverages ARM64 NEON vector instructions and the Apple Accelerate framework (`vDSP`, `cblas`) for high-throughput vectorized linear algebra and autograd computation.

Titan-LM features an **infinite memory-flat streaming tokenizer** capable of ingesting massive, multi-gigabyte codebases with an $O(1)$ flat memory footprint, paired with an out-of-core zero-copy memory-mapped dataset pipeline.

---

## Architectural Highlights

```
Raw Codebases (LLVM, Linux, OpenCV, Qt5, TensorFlow)
                    │
                    ▼  [Recursive Directory Scanner (O(1) RAM)]
      ┌───────────────────────────┐
      │  BPE Streaming Tokenizer  │ ──► titan_vocab.txt (4,096 merges)
      └───────────────────────────┘
                    │
                    ▼  [Direct uint32_t disk streaming]
           titan_tokens.bin (1.15 Billion Tokens / 4.3 GB)
                    │
                    ▼  [POSIX mmap() Zero-Copy]
      ┌───────────────────────────┐
      │ Titan-LM Transformer Core │ ◄── Apple Accelerate (vDSP / cblas)
      │  • 12 Transformer Layers  │ ◄── ARM64 NEON SIMD
      │  • 8 Attention Heads      │
      │  • d_model = 512 (42.15M) │
      └───────────────────────────┘
                    │
         ┌──────────┴──────────┐
         ▼                     ▼
┌──────────────────┐  ┌──────────────────────────┐
│  AdamW Optimizer │  │ Autoregressive Generator │
│  Cosine Decay LR │  │ Top-P Nucleus Sampling   │
│  Weight Decay    │  │ Repetition Penalty (Logit│
└──────────────────┘  └──────────────────────────┘
```

- **Bare-Metal Autograd Engine**: Custom dynamic tape-based reverse-mode automatic differentiation supporting broadcasted tensor operations, fused softmax cross-entropy, and contiguous gradient accumulation.
- **Causal Multi-Head Self-Attention (MHA)**: Vectorized scaled dot-product attention with strict lower-triangular causal masking and multi-head projection layers.
- **Infinite Memory-Flat Streaming Tokenizer**: Avoids intermediate multi-gigabyte text files. Recursively traverses directories, tokenizing file-by-file into an append-only binary buffer with an $O(1)$ memory footprint (< 30 MB RAM) at over **4.85 million tokens/sec**.
- **Zero-Copy Memory-Mapped Dataset**: Maps multi-gigabyte binary token files (`titan_tokens.bin`) directly into memory via `mmap()`, allowing training on datasets that exceed physical RAM capacity.
- **AdamW Optimizer with Cosine Annealing**: Implements decoupled weight decay, first- and second-moment gradient tracking with bias correction, and a cosine learning rate decay schedule with linear warmup.
- **Inference with Top-P & Repetition Penalty**: Real-time autoregressive generation engine with dynamic temperature scaling, Top-P (nucleus) sampling, and frequency-based repetition penalty to prevent repetitive loops.

---

## Model Specifications

| Parameter | Value |
|---|---|
| **Parameters** | **`42,155,008` (42.15M)** |
| **Layers (`n_layers`)** | `12` |
| **Hidden Dimension (`d_model`)** | `512` |
| **Attention Heads (`n_heads`)** | `8` (head dimension: `64`) |
| **Context Window (`max_seq`)** | `256` tokens |
| **Vocabulary Size** | `4,096` custom code-optimized BPE merges |
| **Training Dataset** | **`1,154,060,209` tokens** (4.3 GB contiguous binary) |
| **Training Convergence** | Loss dropped from `8.4081` → **`1.8201`** across 10,000 steps |
| **Hardware Target** | Apple Silicon M4 (10 Execution Threads, AMX + NEON) |

---

## Production Dataset Milestone

Titan-LM was trained on a contiguous **1.154 Billion Token** corpus compiled from major open-source systems engineering codebases:

- **`llvm/llvm-project`**: Modern template C++, compiler passes, and AST architectures.
- **`torvalds/linux`**: Low-level kernel subsystems, memory managers, and hardware drivers.
- **`opencv/opencv`**: High-performance linear algebra, SIMD kernels, and matrix operations.
- **`qt/qt5`**: Declarative QML layouts, meta-object systems, and UI components.
- **`tensorflow/tensorflow`**: C++ computation graphs and operator implementations.

### Ingestion Metrics
- **Files Ingested**: `170,963` source files
- **Raw Code Read**: `2.39 GB`
- **Contiguous Tokens Generated**: `1,154,060,209`
- **Compression Density**: `2.17 bytes/token`
- **Ingestion Speed**: `4.85 Million tokens/sec` (238 seconds wall time)

---

## Directory Structure

```
titan-lm/
├── main.cpp                # CLI entry point, benchmark harness, and subcommands
├── tokenizer.hpp/.cpp      # BPE tokenizer, trie lookup, and recursive streaming pipeline
├── tensor.hpp/.cpp         # N-dimensional tensor engine, autograd tape, and BLAS kernels
├── attention.hpp/.cpp      # Causal Multi-Head Attention and projection layers
├── model.hpp/.cpp          # TitanModel architecture, LayerNorm, and autoregressive generation
├── trainer.hpp/.cpp        # AdamW optimizer, cosine schedule, and mmap dataset loader
├── harvest_huge_corpus.sh  # Automated shallow cloning script for mega-corpora
├── ingest.sh               # Infinite streaming ingestion automation script
├── Makefile                # M4 native clang++ build configuration
├── titan_vocab.txt         # 4,096-token trained BPE vocabulary map
├── titan_weights.bin       # Saved model checkpoint weights (Step 10,000)
└── titan_tokens.bin        # Memory-mapped binary token dataset (1.15B tokens)
```

---

## Quickstart

### 1. Prerequisites
- macOS on Apple Silicon (M1, M2, M3, or M4 recommended)
- Xcode Command Line Tools (`clang++` with C++20 support)
- Git

### 2. Build the Engine
Compile the native binary with Apple Silicon M4 architecture flags and the Accelerate framework:

```bash
git clone https://github.com/GitGuru29/titan-lm.git
cd titan-lm
make
```

### 3. Verify the Architecture
Run the built-in phase verification suite to validate Accelerate BLAS dot-products, autograd backpropagation, attention causal masks, and feed-forward blocks:

```bash
./titan_lm
```

---

## Dataset Ingestion & Training

### 1. Harvest Open-Source Codebases (Optional)
Clone shallow copies of production C++ repositories into `~/titan_massive_dataset`:

```bash
chmod +x harvest_huge_corpus.sh
./harvest_huge_corpus.sh
```

### 2. Run Memory-Flat Streaming Ingestion
Stream and tokenize files directly into `titan_tokens.bin`:

```bash
make ingest
```

### 3. Start the AdamW Training Loop
Launch the training run on the memory-mapped token corpus:

```bash
make train
```

### 4. Track Live Training Progress
Monitor loss, learning rate, and processing speed in real-time from a separate terminal:

```bash
tail -f live_training.log
```

---

## Text & Code Generation

Generate code autoregressively from a prompt using the saved checkpoint:

```bash
# Basic generation:
./titan_lm generate "int main() {"

# Custom length, temperature, and repetition penalty:
./titan_lm generate "template <typename T>" 128 0.7 1.30
./titan_lm generate "static int __init" 128 0.7 1.30
```

### CLI Arguments
```bash
./titan_lm generate <prompt> [max_tokens] [temperature] [repetition_penalty]
```
- `prompt`: The starting context string.
- `max_tokens` *(default: 256)*: Maximum number of tokens to autoregressively decode.
- `temperature` *(default: 0.8)*: Sampling randomness factor.
- `repetition_penalty` *(default: 1.20)*: Multiplicative logit penalty for previously generated tokens to suppress repetition loops.

---

## Generation Samples (Trained Checkpoint)

Here are raw outputs produced by the trained 42.15M model:

#### LLVM License & Header (`template <typename T>`)
```cpp
//===----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===--------------===//
//
// UNY: Apache https://llvm>

#include "majects.h>
#include <linux/type.h>

USTConable.h"
```

#### Linux Kernel Driver Initialization (`static int __init`)
```c
(void, struct *info *si->enc = absl::Parent(char *ctx, char *sim)
{
	struct clk_attribute_device *dev = us->bi_map;
	struct ilio_data context;
};

/*
 * If for a set to disable the driver to drach the error, encd is to eripe in handle too as execution.
 */

#ifndef EB_PIP_RS(n & netdevk) by w
```

---

## Performance Benchmarks

Measured on **Apple Silicon M4** (MacBook Air, 16 GB Unified Memory):

| Component | Benchmark Metric | Result |
|---|---|---|
| **BPE Streaming Ingestion** | File-by-file tokenization | **4.85 Million tokens/sec** |
| **RAM Footprint (Ingestion)** | Working memory during 170K file stream | **< 30 MB (Constant $O(1)$)** |
| **Training Throughput** | 42.15M model forward + backward + AdamW | **~2,060 tokens/sec** |
| **Peak Training Speed** | Micro-batch compute on M4 AMX/NEON | **2,174.8 tokens/sec** |
| **Generation Latency** | Autoregressive Top-P token generation | **~35 ms / token** |

---

## License

This project is licensed under the [MIT License](LICENSE).

