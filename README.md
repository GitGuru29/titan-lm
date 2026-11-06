# Titan-LM

<p align="center">
  <b>A 456M-Parameter Bare-Metal Transformer Engine Written from Scratch in Pure C++20 for Apple Silicon M4</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/C%2B%2B-20-00599C?logo=c%2B%2B" alt="C++20">
  <img src="https://img.shields.io/badge/Hardware-Apple%20Silicon%20M4%20(ARM64)-black?logo=apple" alt="Apple Silicon M4">
  <img src="https://img.shields.io/badge/Acceleration-Apple%20Accelerate%20(vDSP%2Fcblas)-green" alt="Apple Accelerate">
  <img src="https://img.shields.io/badge/Dependencies-Zero%20External%20Libs-orange" alt="Zero Dependencies">
  <img src="https://img.shields.io/badge/Model%20Scale-456M%20Parameters-purple" alt="456M Parameters">
  <img src="https://img.shields.io/badge/License-MIT-blue" alt="License">
</p>

---

## Overview

**Titan-LM** is an end-to-end, high-performance Transformer Large Language Model implemented entirely from first principles in pure C++20. It contains **zero external machine learning dependencies**—no PyTorch, libtorch, GGML, ONNX, or Python runtime.

Engineered specifically for **Apple Silicon (M4 Native)**, Titan-LM leverages ARM64 NEON vector instructions, the Apple Matrix Coprocessor (AMX), and the Apple Accelerate framework (`vDSP`, `cblas`) for high-throughput vectorized linear algebra and autograd computation.

Titan-LM features an **infinite memory-flat streaming tokenizer** capable of ingesting massive multi-gigabyte codebases with an $O(1)$ flat memory footprint, paired with an out-of-core zero-copy memory-mapped dataset pipeline and a turn-aligned Supervised Fine-Tuning (SFT) alignment engine with prompt loss masking.

---

## Architectural Highlights

```
32 Tier-1 Mega-Codebases (Linux, FreeBSD, LLVM, GCC, DuckDB, RocksDB, OpenZFS, Envoy, Postgres...)
                                        │
                                        ▼  [Recursive Scanner (O(1) Constant RAM)]
                          ┌───────────────────────────┐
                          │  BPE Streaming Tokenizer  │ ──► titan_vocab.txt (4,096 merges)
                          └───────────────────────────┘
                                        │
                                        ▼  [Direct uint32_t disk streaming @ 6.5M tok/s]
                             titan_tokens.bin (2.197 Billion Tokens / 8.38 GB)
                                        │
                                        ▼  [POSIX mmap() Zero-Copy Page Cache]
                          ┌───────────────────────────┐
                          │ Titan-450M Transformer    │ ◄── Apple Accelerate (vDSP / cblas_sgemm)
                          │  • 28 Transformer Layers  │ ◄── Apple Matrix Coprocessor (AMX)
                          │  • 16 Attention Heads     │ ◄── ARM64 NEON Vector Extensions
                          │  • d_model = 1152         │
                          │  • 456,060,672 Parameters │
                          └───────────────────────────┘
                                        │
                     ┌──────────────────┴──────────────────┐
                     ▼                                     ▼
      ┌─────────────────────────────┐       ┌─────────────────────────────┐
      │      Pre-Training Core      │       │     Turn-Aligned SFT Core   │
      │  • AdamW (Cosine Schedule)  │       │  • Prompt Target Masking    │
      │  • Decoupled Weight Decay   │       │  • Pure C/C++ & General QA  │
      │  • 10,000 Continuous Steps  │       │  • 5,000 Alignment Steps   │
      └─────────────────────────────┘       └─────────────────────────────┘
                     │                                     │
                     └──────────────────┬──────────────────┘
                                        ▼
                         ┌─────────────────────────────┐
                         │   Conversational Dialogue   │
                         │  • Top-P Nucleus Sampling   │
                         │  • Multi-Token Stop Masks   │
                         │  • Repetition Penalty       │
                         └─────────────────────────────┘
```

- **Bare-Metal Autograd Engine**: Custom dynamic tape-based reverse-mode automatic differentiation supporting broadcasted tensor operations, fused softmax cross-entropy, and contiguous gradient accumulation.
- **Titan-450M Scaled Architecture**: 28 deep Pre-LN residual transformer blocks with 16-head scaled causal attention, utilizing ~13.8 GB unified memory on 16 GB Apple Silicon Macs.
- **Turn-Aligned SFT with Prompt Target Loss Masking**: Autoregressive conversational alignment masking prompt tokens with `IGNORE_INDEX = 0xFFFFFFFF`, preventing question hallucination and ensuring loss gradients update only on assistant responses.
- **Infinite Memory-Flat Streaming Tokenizer**: Recursively tokenizes directories into contiguous binary token arrays with an $O(1)$ memory footprint (< 30 MB RAM) at over **6.5 million tokens/sec**.
- **Zero-Copy Memory-Mapped Dataset**: Maps multi-gigabyte binary token files (`titan_tokens.bin`) directly into memory via POSIX `mmap()`, allowing seamless streaming across billions of tokens without physical RAM exhaustion.
- **Inference with Top-P & Multi-Token Stop Suppression**: Real-time autoregressive generation engine with dynamic temperature scaling, Top-P (nucleus) sampling, frequency-based repetition penalty, and retroactive multi-token stop sequence suppression.

---

## Model Specifications

| Parameter | Titan-450M (Production) | Titan-42M (Compact) |
|---|---|---|
| **Parameters** | **`456,060,672` (456.06M)** | `42,155,008` (42.15M) |
| **Hidden Dimension (`d_model`)** | `1,152` | `512` |
| **Layers (`n_layers`)** | `28` | `12` |
| **Attention Heads (`n_heads`)** | `16` (head dimension: `72`) | `8` (head dimension: `64`) |
| **Context Window (`max_seq`)** | `256` tokens | `256` tokens |
| **Vocabulary Size** | `4,096` code-optimized BPE merges | `4,096` code-optimized BPE merges |
| **Training Dataset** | **`2,197,769,052` tokens** (8.38 GB binary) | `1,154,060,209` tokens (4.3 GB binary) |
| **Pre-Training Loss** | `8.5122` → **`2.4706`** (10,000 steps) | `8.4081` → `1.8201` (10,000 steps) |
| **SFT Alignment Loss** | `4.7367` → **`1.1841`** (5,000 steps) | — |
| **Unified RAM Allocation** | **~13.8 GB / 16 GB** | ~3.4 GB / 16 GB |
| **Hardware Target** | Apple Silicon M4 (AMX + NEON + Accelerate) | Apple Silicon M4 |

---

## 32-Codebase Mega-Corpus Dataset

Titan-450M was pre-trained on a contiguous **2.197 Billion Token** corpus harvested across 32 premier Tier-1 open-source systems engineering repositories:

- **Operating Systems & Kernels**: `torvalds/linux`, `freebsd/freebsd-src`, `serenityos/serenity`, `haiku/haiku`, `reactos/reactos`.
- **Compilers & Runtimes**: `llvm/llvm-project`, `gcc-mirror/gcc`, `tree-sitter/tree-sitter`.
- **Databases & High-Throughput Engines**: `duckdb/duckdb`, `facebook/rocksdb`, `postgres/postgres`, `openzfs/zfs`.
- **Distributed Systems & Networking**: `envoyproxy/envoy`, `grpc/grpc`, `redis/redis`.
- **Core Systems Infrastructure & Concurrency**: `oneapi-src/oneTBB`, `facebook/folly`, `abseil/abseil-cpp`, `openssl/openssl`, `FFmpeg/FFmpeg`, `hyprwm/Hyprland`, `Kitware/CMake`, `ggerganov/llama.cpp`.

### Ingestion Metrics
- **Files Ingested**: `406,139` source files
- **Raw Code Read**: `4.63 GB`
- **Contiguous Tokens Generated**: `2,197,769,052` tokens
- **Binary Dataset Size**: `8.38 GB` (`titan_tokens.bin`)
- **Ingestion Speed**: `6.54 Million tokens/sec`

---

## Conversational SFT Alignment Pipeline

To transform the base completion model into an aligned conversational AI assistant, Titan-LM employs an aligned SFT pipeline with prompt loss masking:

1. **Pure C/C++ Instruction Extraction**: Harvested from high-density instruction splits, strictly excluding Python, JavaScript, and dynamic language patterns to enforce 100% pure C and C++ syntax generation.
2. **General Knowledge & Reasoning**: Cleaned conversational question-answering pairs covering technical explanations, logic, and general knowledge.
3. **Reinforced AI Persona & Greetings**: Embedded dialogue turns ensuring natural, coherent conversational greetings (`"Hi"`, `"Who are you?"`) and zero jargon hallucination.
4. **Target Loss Masking (`IGNORE_INDEX = 0xFFFFFFFF`)**:
   $$\mathcal{L} = -\frac{1}{\sum_{t} \mathbb{I}(y_t \ne \text{IGNORE})} \sum_{t, y_t \ne \text{IGNORE}} \log P(y_t \mid x_{<t})$$
   Gradients are calculated exclusively on assistant response tokens, eliminating autoregressive prompt echoing.

---

## Quickstart

### 1. Prerequisites
- macOS on Apple Silicon (M1, M2, M3, or M4 recommended)
- Xcode Command Line Tools (`clang++` with C++20 support)
- 16 GB Unified Memory (for Titan-450M)

### 2. Build the Engine
Compile the native binary with Apple Silicon M4 architecture flags and the Accelerate framework:

```bash
git clone https://github.com/GitGuru29/titan-lm.git
cd titan-lm
make
```

### 3. Verify the Architecture
Run the built-in phase verification suite to validate Accelerate BLAS matrix multiplications, autograd backpropagation, attention causal masks, and feed-forward blocks:

```bash
./titan_lm
```

---

## Interactive Chat & Text Generation

### Conversational Dialogue (SFT Checkpoint)
Interact with the aligned Titan-450M assistant:

```bash
# General greeting:
./titan_lm chat "Hi"

# General knowledge:
./titan_lm chat "What are the primary colors?"

# Systems & C++ Programming:
./titan_lm chat "Write a C++ function to add two numbers"
./titan_lm chat "Explain how a thread-safe circular buffer works in C++"
```

### Raw Autoregressive Code Completion (Base Checkpoint)
Generate code directly from a prompt using the pre-trained base model:

```bash
# Basic code generation:
./titan_lm generate "template <typename T> class Vector {"

# Custom length, temperature, and repetition penalty:
./titan_lm generate "static int __init" 128 0.7 1.30
```

---

## Performance Benchmarks

Measured on **Apple Silicon M4** (MacBook, 16 GB Unified Memory):

| Component | Benchmark Metric | Result |
|---|---|---|
| **BPE Streaming Ingestion** | File-by-file tokenization | **6.54 Million tokens/sec** |
| **RAM Footprint (Ingestion)** | Working memory during 406K file stream | **< 30 MB (Constant $O(1)$)** |
| **Pre-Training Throughput** | 456M model forward + backward + AdamW | **~342 tokens/sec** |
| **Inference Latency** | Autoregressive Top-P token generation | **~18 ms / token** |
| **Unified Memory Footprint** | Model weights + Gradients + AdamW state | **7.30 GB** |
| **Total System RAM Allocation** | Working Set + Activation Cache + mmap | **~13.8 GB / 16 GB** |

---

## Branches

- **`main`**: Clean, high-throughput Titan-450M engine with base pre-training, turn-aligned SFT alignment, and conversational chat inference.
- **`feature/thermal-governor`**: Includes the **Smart Thermal & Battery Governor**—an in-process duty-cycle manager running 30-min training / 10-min in-place cooling breaks with macOS battery and die thermal health checks.

---

## License

This project is licensed under the [MIT License](LICENSE).
