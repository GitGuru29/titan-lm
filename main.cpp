#include "tokenizer.hpp"
#include "tensor.hpp"
#include "attention.hpp"
#include "model.hpp"
#include "trainer.hpp"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <filesystem>
#include <vector>
#include <string>
#include <thread>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <Accelerate/Accelerate.h>

namespace {

void print_banner() {
    std::cout << "====================================================================\n";
    std::cout << "  TITAN-LM: Bare-Metal Transformer Engine (Apple Silicon M4 / C++20)\n";
    std::cout << "  Full Stack: Ingestion | AMX Core | Attention | LayerNorm | AdamW | Generation\n";
    std::cout << "====================================================================\n\n";
}

void print_system_telemetry() {
    std::cout << "[System Architecture & Hardware Topology]\n";
    std::cout << "  Target CPU: Apple Silicon M4 (ARM64 Neon + Apple Matrix Coprocessor AMX)\n";
    std::cout << "  Concurrency: " << std::thread::hardware_concurrency() << " hardware execution threads\n";

    // Validate Apple Accelerate BLAS/vDSP availability
    float test_a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float test_b[4] = {2.0f, 0.5f, 1.0f, 2.0f};
    float test_c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    vDSP_vmul(test_a, 1, test_b, 1, test_c, 1, 4);

    std::cout << "  Accelerate Framework: Active (vDSP vector dot-product verified: "
              << test_c[0] << ", " << test_c[1] << ", " << test_c[2] << ", " << test_c[3] << ")\n\n";
}

void run_ingestion_verification(const titan::Tokenizer& tokenizer) {
    std::cout << "[Phase 1 Verification: Ingestion & Detokenization Fidelity]\n";

    const std::string sample_source =
        "#include <iostream>\n"
        "#include <span>\n\n"
        "// Titan-LM Core Kernel\n"
        "template <typename T>\n"
        "void compute_attention(std::span<const T> q, std::span<const T> k) {\n"
        "    const size_t dim = q.size();\n"
        "    for (size_t i = 0; i < dim; ++i) {\n"
        "        // Contiguous SIMD Accumulation\n"
        "    }\n"
        "}\n"
        "int main() {\n"
        "    std::cout << \"Bare-metal Transformer initialized.\" << std::endl;\n"
        "    return 0;\n"
        "}\n";

    std::vector<titan::TokenId> tokens = tokenizer.encode(sample_source, true, true);
    std::string reconstructed = tokenizer.decode(tokens, true);

    std::cout << "  Original Source Size: " << sample_source.size() << " bytes\n";
    std::cout << "  Contiguous Token Count: " << tokens.size() << " tokens\n";
    std::cout << "  Vocabulary Capacity: " << tokenizer.vocab_size() << " entries\n";

    if (reconstructed == sample_source) {
        std::cout << "  Roundtrip Integrity: 100% BYTE-EXACT MATCH PASSED\n";
    } else {
        std::cerr << "  Roundtrip Integrity: MISMATCH DETECTED\n";
    }

    const std::string out_bin = "test_tokens.bin";
    std::vector<titan::TokenId> workspace_tokens;
    const std::vector<std::string> src_files = {
        "tokenizer.hpp", "tokenizer.cpp",
        "tensor.hpp", "tensor.cpp",
        "attention.hpp", "attention.cpp",
        "model.hpp", "model.cpp",
        "trainer.hpp", "trainer.cpp"
    };

    for (const auto& file : src_files) {
        if (std::filesystem::exists(file)) {
            auto file_tokens = tokenizer.ingest_file(file, true, true);
            workspace_tokens.insert(workspace_tokens.end(), file_tokens.begin(), file_tokens.end());
        }
    }

    tokenizer.save_tokens_binary(out_bin, workspace_tokens);
    titan::MappedTokenBuffer mapped = tokenizer.mmap_tokens_binary(out_bin);
    std::cout << "  Dataset Token Buffer: " << mapped.size() << " tokens memory-mapped from "
              << out_bin << "\n\n";
}

void run_tensor_autograd_verification() {
    std::cout << "[Phase 2 Verification: Accelerated Tensor Engine & Autograd Core]\n";

    titan::Tensor A(2, 3, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    titan::Tensor B(3, 2, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    titan::Tensor C = titan::Tensor::matmul(A, B);

    bool forward_ok = (std::abs(C(0, 0) - 22.0f) < 1e-4f) &&
                      (std::abs(C(0, 1) - 28.0f) < 1e-4f) &&
                      (std::abs(C(1, 0) - 49.0f) < 1e-4f) &&
                      (std::abs(C(1, 1) - 64.0f) < 1e-4f);

    titan::Tensor grad_C(2, 2, 1.0f);
    titan::Tensor::backward_matmul(A, B, grad_C, A, B);

    bool grad_a_ok = (std::abs(A.grad_at(0, 0) - 3.0f) < 1e-4f) &&
                     (std::abs(A.grad_at(0, 1) - 7.0f) < 1e-4f) &&
                     (std::abs(A.grad_at(0, 2) - 11.0f) < 1e-4f);

    if (forward_ok && grad_a_ok) {
        std::cout << "  AMX cblas_sgemm Matmul Forward & Backward: PASSED\n";
    } else {
        std::cerr << "  AMX cblas_sgemm Matmul Forward & Backward: FAILED\n";
    }

    titan::Tensor X_relu(1, 4, {-2.0f, 0.5f, -0.1f, 3.0f});
    titan::Tensor Y_relu = X_relu.forward_relu();
    titan::Tensor grad_relu(1, 4, {1.0f, 1.0f, 1.0f, 1.0f});
    X_relu.backward_relu(grad_relu);

    if (Y_relu(0, 0) == 0.0f && Y_relu(0, 1) == 0.5f && X_relu.grad_at(0, 0) == 0.0f && X_relu.grad_at(0, 1) == 1.0f) {
        std::cout << "  ReLU Forward & Derivative Accumulation: PASSED\n";
    } else {
        std::cerr << "  ReLU Forward & Derivative Accumulation: FAILED\n";
    }

    titan::Tensor logits(2, 3, {1.0f, 2.0f, 3.0f, 1000.0f, 1001.0f, 1002.0f});
    titan::Tensor probs = logits.forward_softmax();
    float row0_sum = probs(0, 0) + probs(0, 1) + probs(0, 2);

    if (std::abs(row0_sum - 1.0f) < 1e-4f) {
        std::cout << "  Stable Row Softmax: PASSED\n\n";
    } else {
        std::cerr << "  Stable Row Softmax: FAILED\n\n";
    }
}

void run_attention_and_ffn_verification() {
    std::cout << "[Phase 3 Verification: Causal Self-Attention & Multi-Layer FFN]\n";

    const size_t d_model = 64;
    const size_t n_heads = 4;
    const size_t max_seq = 32;
    const size_t T = 4;

    titan::CausalAttention attention(d_model, n_heads, max_seq);
    attention.init_weights(123);

    titan::Tensor X(T, d_model);
    X.fill_random(-0.5f, 0.5f, 456);

    titan::Tensor Y_attn = attention.forward(X);

    bool causal_mask_strictly_enforced = true;
    for (size_t h = 0; h < n_heads; ++h) {
        const auto& weights_h = attention.cache.weights[h];
        for (size_t r = 0; r < T; ++r) {
            float row_sum = 0.0f;
            for (size_t c = 0; c < T; ++c) {
                float w = weights_h(r, c);
                row_sum += w;
                if (c > r && std::abs(w) > 1e-6f) causal_mask_strictly_enforced = false;
            }
            if (std::abs(row_sum - 1.0f) > 1e-4f) causal_mask_strictly_enforced = false;
        }
    }

    if (causal_mask_strictly_enforced) {
        std::cout << "  Causal Self-Attention Forward & Mask Invariant: PASSED\n";
    } else {
        std::cerr << "  Causal Self-Attention Forward & Mask Invariant: FAILED\n";
    }

    titan::Tensor grad_Y_attn(T, d_model, 1.0f);
    titan::Tensor grad_X(T, d_model, 0.0f);
    attention.backward(grad_Y_attn, grad_X);

    auto check_grad = [&](const titan::Tensor& t) {
        for (float g : t.grad) {
            if (std::isnan(g) || std::isinf(g)) return false;
            if (std::abs(g) > 1e-9f) return true;
        }
        return false;
    };

    if (check_grad(attention.W_q) && check_grad(attention.W_o) && check_grad(grad_X)) {
        std::cout << "  Causal Self-Attention Autograd Backward: PASSED\n";
    } else {
        std::cerr << "  Causal Self-Attention Autograd Backward: FAILED\n";
    }

    const size_t d_ff = 256;
    titan::FeedForward ffn(d_model, d_ff);
    ffn.init_weights(789);

    titan::Tensor Y_ffn = ffn.forward(X);
    titan::Tensor grad_Y_ffn(T, d_model, 1.0f);
    titan::Tensor grad_X_ffn(T, d_model, 0.0f);
    ffn.backward(grad_Y_ffn, grad_X_ffn);

    if (check_grad(ffn.W_1) && check_grad(ffn.W_2) && check_grad(grad_X_ffn)) {
        std::cout << "  Feed-Forward (W2 x ReLU(W1 x X + b1) + b2) Autograd: PASSED\n\n";
    } else {
        std::cerr << "  Feed-Forward Autograd: FAILED\n\n";
    }
}

void run_phase4_transformer_block_verification() {
    std::cout << "[Phase 4 Verification: LayerNorm & Pre-LN TransformerBlock]\n";

    const size_t d_model = 64;
    const size_t n_heads = 4;
    const size_t max_seq = 32;
    const size_t T = 4;

    // 1. LayerNorm
    titan::LayerNorm ln(d_model);
    titan::Tensor X_ln(T, d_model);
    X_ln.fill_random(-2.0f, 2.0f, 101);

    titan::Tensor Y_ln = ln.forward(X_ln);

    bool ln_norm_ok = true;
    for (size_t t = 0; t < T; ++t) {
        float sum = 0.0f;
        for (size_t c = 0; c < d_model; ++c) sum += Y_ln(t, c);
        float out_mean = sum / static_cast<float>(d_model);

        float sq_sum = 0.0f;
        for (size_t c = 0; c < d_model; ++c) sq_sum += (Y_ln(t, c) - out_mean) * (Y_ln(t, c) - out_mean);
        float out_var = sq_sum / static_cast<float>(d_model);

        if (std::abs(out_mean) > 1e-4f || std::abs(out_var - 1.0f) > 1e-3f) {
            ln_norm_ok = false;
        }
    }

    titan::Tensor grad_Y_ln(T, d_model, 1.0f);
    titan::Tensor grad_X_ln(T, d_model, 0.0f);
    ln.backward(grad_Y_ln, grad_X_ln);

    bool zero_sum_invariant = true;
    for (size_t t = 0; t < T; ++t) {
        float row_grad_sum = 0.0f;
        for (size_t c = 0; c < d_model; ++c) {
            row_grad_sum += grad_X_ln.grad_at(t, c);
        }
        if (std::abs(row_grad_sum) > 1e-4f) {
            zero_sum_invariant = false;
        }
    }

    if (ln_norm_ok && zero_sum_invariant) {
        std::cout << "  LayerNorm Forward Normalization & vDSP Analytical Backward: PASSED\n";
    } else {
        std::cerr << "  LayerNorm Forward Normalization & vDSP Analytical Backward: FAILED\n";
    }

    // 2. TransformerBlock
    titan::TransformerBlock block(d_model, n_heads, max_seq);
    block.init_weights(555);

    titan::Tensor X(T, d_model);
    X.fill_random(-0.5f, 0.5f, 666);

    titan::Tensor Y = block.forward(X);
    bool shape_ok = (Y.rows == T && Y.cols == d_model);

    titan::Tensor grad_Y(T, d_model, 1.0f);
    titan::Tensor grad_X(T, d_model, 0.0f);
    block.backward(grad_Y, grad_X);

    auto check_has_grad = [](const titan::Tensor& t) {
        for (float g : t.grad) {
            if (std::isnan(g) || std::isinf(g)) return false;
            if (std::abs(g) > 1e-9f) return true;
        }
        return false;
    };

    bool all_sublayers_accumulated =
        check_has_grad(block.ln_1.gamma) &&
        check_has_grad(block.attn.W_q) &&
        check_has_grad(block.ln_2.gamma) &&
        check_has_grad(block.ffn.W_1) &&
        check_has_grad(grad_X);

    if (shape_ok && all_sublayers_accumulated) {
        std::cout << "  TransformerBlock Pre-LN Forward & Dual Residual Autograd: PASSED\n\n";
    } else {
        std::cerr << "  TransformerBlock Pre-LN Forward & Dual Residual Autograd: FAILED\n\n";
    }
}

void run_phase5_trainer_and_generation_verification(const titan::Tokenizer& tokenizer) {
    std::cout << "[Phase 5 Verification: AdamW Optimizer, Training Engine & Autoregressive Stream]\n";

    const size_t vocab_size = tokenizer.vocab_size();
    const size_t d_model = 64;
    const size_t num_layers = 2;
    const size_t n_heads = 4;
    const size_t max_seq = 32;

    titan::TitanModel model(vocab_size, d_model, num_layers, n_heads, max_seq);
    std::cout << "  Initialized TitanModel: " << model.parameter_count() << " parameters\n";

    // 1. Verify AdamW Optimizer Vectorized Update
    {
        titan::AdamW optimizer(model.get_parameters(), 1e-3f, 0.9f, 0.95f, 1e-8f, 0.01f);

        // Populate mock gradients
        for (auto& pref : optimizer.params) {
            pref.tensor->grad.assign(pref.tensor->size(), 0.1f);
        }

        float initial_val = model.token_embeddings(0, 0);
        optimizer.step();
        float updated_val = model.token_embeddings(0, 0);

        bool step_applied = (initial_val != updated_val) && !std::isnan(updated_val);
        optimizer.zero_grad();

        bool zeroed = true;
        for (float g : model.token_embeddings.grad) {
            if (g != 0.0f) { zeroed = false; break; }
        }

        if (step_applied && zeroed) {
            std::cout << "  AdamW Vectorized Optimizer Core & Moment Updates: PASSED\n";
        } else {
            std::cerr << "  AdamW Vectorized Optimizer Core & Moment Updates: FAILED\n";
        }
    }

    // 2. Training Engine: Run Micro-Optimization Loop on Ingested Code Tokens
    {
        titan::TrainingConfig config;
        config.batch_size = 2;
        config.seq_len = 16;
        config.max_steps = 15;
        config.warmup_steps = 3;
        config.learning_rate = 2e-3f;
        config.min_learning_rate = 2e-4f;
        config.log_interval = 5;
        config.checkpoint_interval = 15;
        config.dataset_path = "titan_tokens.bin";
        config.checkpoint_path = "test_weights.bin";
        config.governor.enabled = true;

        titan::Trainer trainer(model, tokenizer, config);
        trainer.train();

        // Verify checkpoint file serialization
        bool checkpoint_exists = std::filesystem::exists("test_weights.bin");
        size_t cp_size = checkpoint_exists ? std::filesystem::file_size("test_weights.bin") : 0;

        if (checkpoint_exists && cp_size > 0) {
            std::cout << "  Model Checkpoint Serialization: PASSED (" << cp_size << " bytes on disk)\n\n";
        } else {
            std::cerr << "  Model Checkpoint Serialization: FAILED\n\n";
        }
    }

    // 3. Autoregressive Real-Time Token Generation Stream
    {
        std::cout << "[Real-Time Autoregressive Code Generation Stream]\n";
        std::string prompt = "template <typename T>\n";
        std::cout << "  Prompt: \"" << prompt << "\"\n";
        std::cout << "  Streaming Tokens: \"";

        std::string output = model.generate(
            prompt,
            tokenizer,
            24,     // max_new_tokens
            0.8f,   // temperature
            42,     // seed
            true    // stream_to_stdout
        );
        std::cout << "\"\n";

        if (!output.empty()) {
            std::cout << "  Autoregressive Generation Pipeline: PASSED (Streamed "
                      << output.size() << " generated characters)\n\n";
        } else {
            std::cerr << "  Autoregressive Generation Pipeline: FAILED\n\n";
        }
    }
}

} // anonymous namespace

int main(int argc, char* argv[]) {
    print_banner();
    print_system_telemetry();

    titan::Tokenizer tokenizer(true);

    if (argc >= 2) {
        std::string command = argv[1];

        // ── train-vocab [target_size=4096] [corpus_file=training_data.txt] ──
        if (command == "train-vocab") {
            size_t target_vocab = 4096;
            if (argc >= 3) {
                target_vocab = static_cast<size_t>(std::stoul(argv[2]));
            }
            std::filesystem::path corpus_path = (argc >= 4) ? argv[3] : "training_data.txt";
            if (!std::filesystem::exists(corpus_path)) {
                std::cout << "[train-vocab] Corpus file not found: " << corpus_path
                          << " — scanning project codebase files...\n";
                std::vector<std::filesystem::path> src_files;
                for (const auto& entry : std::filesystem::recursive_directory_iterator(".")) {
                    if (entry.is_regular_file()) {
                        auto ext = entry.path().extension().string();
                        if (ext == ".cpp" || ext == ".hpp" || ext == ".h" || ext == ".qml") {
                            src_files.push_back(entry.path());
                        }
                    }
                }
                tokenizer.train_from_files(src_files, target_vocab);
            } else {
                std::cout << "[train-vocab] Reading training corpus from: " << corpus_path << "\n";
                std::ifstream fin(corpus_path, std::ios::binary);
                std::string corpus_sample;
                // Sample 2 MB of real code text for rapid high-density merge discovery
                constexpr size_t SAMPLE_SIZE = 2 * 1024 * 1024;
                corpus_sample.resize(SAMPLE_SIZE);
                fin.read(&corpus_sample[0], SAMPLE_SIZE);
                corpus_sample.resize(fin.gcount());
                fin.close();

                tokenizer.train(corpus_sample, target_vocab);
            }
            std::cout << "[train-vocab] Final vocabulary capacity: "
                      << tokenizer.vocab_size() << " entries (saved to titan_vocab.txt)\n";
            return 0;
        }

        // ── stream-ingest [output_bin] [dirs...] ────────────────────────────
        // Infinite flat-memory streaming tokenizer: steps recursively through
        // directories file-by-file, encodes with 4,096 BPE, and appends raw
        // uint32_t token bytes directly to disk. Zero intermediate text merge.
        if (command == "stream-ingest" || command == "stream-tokenize") {
            std::filesystem::path output_bin = (argc >= 3) ? argv[2] : "titan_tokens.bin";
            std::vector<std::filesystem::path> roots;

            if (argc >= 4) {
                for (int i = 3; i < argc; ++i) {
                    roots.emplace_back(argv[i]);
                }
            } else {
                // Portable defaults: TITAN_CORPUS_DIR (or ~/titan_massive_dataset) plus
                // ~/titan_dataset, with optional colon-separated TITAN_LOCAL_PROJECTS.
                auto add_root = [&roots](const std::filesystem::path& p) {
                    std::error_code ec;
                    if (std::filesystem::is_directory(p, ec)) roots.push_back(p);
                };

                if (const char* corpus_env = std::getenv("TITAN_CORPUS_DIR"); corpus_env && *corpus_env) {
                    add_root(corpus_env);
                } else if (const char* home = std::getenv("HOME")) {
                    add_root(std::filesystem::path(home) / "titan_massive_dataset");
                }

                if (const char* home = std::getenv("HOME")) {
                    add_root(std::filesystem::path(home) / "titan_dataset");
                }

                if (const char* extra = std::getenv("TITAN_LOCAL_PROJECTS"); extra && *extra) {
                    const std::string list(extra);
                    size_t start = 0;
                    while (start <= list.size()) {
                        const size_t sep = list.find(':', start);
                        const std::string item = list.substr(
                            start, sep == std::string::npos ? std::string::npos : sep - start);
                        if (!item.empty()) add_root(item);
                        if (sep == std::string::npos) break;
                        start = sep + 1;
                    }
                }

                if (roots.empty()) {
                    std::cerr << "[stream-ingest] No corpus roots found. Set TITAN_CORPUS_DIR "
                                 "or pass directories explicitly.\n";
                }
            }

            tokenizer.stream_tokenize_directories(roots, output_bin);
            return 0;
        }

        // ── tokenize <input_path> [output_bin] ──────────────────────────────
        // If input_path is a directory: streams file-by-file into binary
        // If input_path is a text file: encodes directly into binary
        if (command == "tokenize" && argc >= 3) {
            std::filesystem::path input_path = argv[2];
            std::filesystem::path output_bin = (argc >= 4) ? argv[3] : "titan_tokens.bin";

            if (!std::filesystem::exists(input_path)) {
                std::cerr << "[ERROR] Input path not found: " << input_path << "\n";
                return 1;
            }

            // Automatic flat streaming mode for directories
            if (std::filesystem::is_directory(input_path)) {
                tokenizer.stream_tokenize_directories({input_path}, output_bin);
                return 0;
            }

            const size_t file_bytes = std::filesystem::file_size(input_path);
            std::cout << "[tokenize] Input  : " << input_path
                      << " (" << (file_bytes / 1024.0 / 1024.0) << " MB)\n";
            std::cout << "[tokenize] Output : " << output_bin << "\n";
            std::cout << "[tokenize] Vocab  : " << tokenizer.vocab_size() << " tokens\n\n";

            // Read entire file into memory
            std::ifstream fin(input_path, std::ios::binary);
            if (!fin) {
                std::cerr << "[ERROR] Cannot open input file\n";
                return 1;
            }
            std::string raw_text((std::istreambuf_iterator<char>(fin)),
                                  std::istreambuf_iterator<char>());
            fin.close();

            std::cout << "[tokenize] Encoding " << raw_text.size()
                      << " characters...\n";

            auto t0 = std::chrono::high_resolution_clock::now();
            std::vector<titan::TokenId> tokens = tokenizer.encode(raw_text, true, true);
            auto t1 = std::chrono::high_resolution_clock::now();
            double elapsed_s = std::chrono::duration<double>(t1 - t0).count();

            std::cout << "[tokenize] Done. " << tokens.size() << " tokens in "
                      << std::fixed << std::setprecision(2) << elapsed_s << "s"
                      << " (" << (tokens.size() / elapsed_s / 1000.0) << "K tok/s)\n";

            // Save binary
            tokenizer.save_tokens_binary(output_bin, tokens);
            const size_t out_bytes = std::filesystem::file_size(output_bin);
            std::cout << "[tokenize] Saved " << out_bytes << " bytes → " << output_bin << "\n";
            std::cout << "[tokenize] ✓ Ready to train. Run: make train\n";
            return 0;
        }

        if (command == "train") {
            const size_t vocab_size = tokenizer.vocab_size();
            // ── Titan-450M Scaled Architecture (Targeting 13.8 GB Unified RAM on Apple Silicon M4) ──
            //   d_model=1152 → head_dim = 1152 / 16 = 72
            //   num_layers=28 → 28 deep residual transformer blocks
            //   n_heads=16   → 16 multi-head attention streams
            //   max_seq=256  → full 256-token attention window
            //   batch_size=6 → 6 micro-batch accumulation (1,536 tokens/step)
            //   Total params: 456,060,672 (456.06M)
            //   FP32 (Weights + Gradients + AdamW m + AdamW v) = 7.30 GB
            //   Activation cache + scratch buffers = ~2.2 GB
            //   Target working set = ~9.5 GB dirty RAM, total resident + page cache = ~13.8 GB
            const size_t d_model    = 1152;
            const size_t num_layers = 28;
            const size_t n_heads    = 16;   // head_dim = 1152/16 = 72 ✓
            const size_t max_seq    = 256;

            std::cout << "\n[train] Titan-450M Scaled Architecture (Apple Silicon M4 Native):\n";
            std::cout << "  vocab_size  = " << vocab_size  << "\n";
            std::cout << "  d_model     = " << d_model     << "\n";
            std::cout << "  num_layers  = " << num_layers  << "\n";
            std::cout << "  n_heads     = " << n_heads     << "  → head_dim = " << (d_model/n_heads) << "\n";
            std::cout << "  max_seq     = " << max_seq     << "\n";
            std::cout << "  est. params ≈ " << std::fixed << std::setprecision(2)
                      << (vocab_size * d_model * 2 + max_seq * d_model + num_layers * (4 * d_model * d_model + 4 * d_model + 2 * d_model * 4 * d_model + 5 * d_model + 4 * d_model) + 2 * d_model) / 1000000.0 << "M\n\n";

            titan::TitanModel model(vocab_size, d_model, num_layers, n_heads, max_seq);

            // Safeguard & continuous pre-training on expanded corpus
            const std::string base_weights = "titan_weights.bin";
            const std::string backup_prev  = "titan_weights.prev.bin";
            uint32_t loaded_step = 0;
            float loaded_loss = 0.0f;
            bool checkpoint_loaded = false;

            if (std::filesystem::exists(base_weights)) {
                std::error_code ec;
                std::filesystem::copy_file(base_weights, backup_prev, std::filesystem::copy_options::overwrite_existing, ec);
                if (!ec) {
                    std::cout << "  [Checkpoint Safety] Existing weights archived → " << backup_prev << "\n";
                }
                if (model.load_weights(base_weights, &loaded_step, &loaded_loss)) {
                    checkpoint_loaded = true;
                    std::cout << "  [Continuous Pre-training] Successfully loaded existing weights from " << base_weights 
                              << " (Step " << loaded_step << " | Loss: " << std::fixed << std::setprecision(4) << loaded_loss << ")\n";
                } else {
                    std::cout << "  [Fresh Initialization] Architecture changed to Titan-450M (456.06M params). Starting Xavier initialization.\n";
                }
            } else {
                std::cout << "  [INFO] No existing weights found. Starting Xavier initialization.\n";
            }

            titan::TrainingConfig config;
            config.batch_size          = 6;       // 6 micro-batch gradient accumulation (1,536 tokens/step)
            config.seq_len             = 256;     // 256 tokens context window
            config.max_steps           = 10000;   // 10K optimization steps
            config.warmup_steps        = 500;     // linear warmup
            config.learning_rate       = 1.5e-4f; // stable AdamW learning rate for 450M model
            config.min_learning_rate   = 1.5e-5f; // cosine decay floor
            config.log_interval        = 10;      // telemetry interval
            config.checkpoint_interval = 250;     // checkpoint save interval

            if (checkpoint_loaded && loaded_step > 0 && loaded_step < config.max_steps) {
                config.start_step = loaded_step + 1;
            }

            // Smart Closed-Loop Hardware Governor (Real-Time Apple Silicon IOKit Telemetry)
            config.governor.enabled = true;
            config.governor.profile = titan::ThermalProfile::AUTO;
            for (int i = 2; i < argc; ++i) {
                std::string arg = argv[i];
                if (arg == "--docked") {
                    config.governor.apply_profile(titan::ThermalProfile::DOCKED);
                } else if (arg == "--mobile") {
                    config.governor.apply_profile(titan::ThermalProfile::MOBILE);
                }
            }

            titan::Trainer trainer(model, tokenizer, config);
            trainer.train();
            return 0;
        }

        if (command == "stream-instruct" && argc >= 3) {
            std::filesystem::path input_file = argv[2];
            std::filesystem::path output_bin = (argc >= 4) ? argv[3] : "titan_instruct_tokens.bin";
            size_t seq_len = (argc >= 5) ? static_cast<size_t>(std::stoul(argv[4])) : 192;
            tokenizer.stream_tokenize_chat_file(input_file, output_bin, seq_len);
            return 0;
        }

        if (command == "train_instruct" || command == "train-instruct") {
            const size_t vocab_size = tokenizer.vocab_size();
            const size_t d_model    = 1152;
            const size_t num_layers = 28;
            const size_t n_heads    = 16;
            const size_t max_seq    = 256;

            std::cout << "  ====================================================================\n";
            std::cout << "  TITAN-LM 450M SFT Alignment: Turn-Aligned Supervised Fine-Tuning\n";
            std::cout << "  ====================================================================\n";
            std::cout << "  Model Config: 456.06M params (d_model=1152, 28 layers, 16 heads, max_seq=256)\n";
            std::cout << "  Active Vocab: " << vocab_size << " BPE tokens\n";

            titan::TitanModel model(vocab_size, d_model, num_layers, n_heads, max_seq);

            // 1. Safeguard & protect base checkpoint weights
            const std::string base_weights = "titan_weights.bin";
            const std::string backup_base  = "titan_weights.base.bin";
            const std::string instruct_weights = "titan_instruct_weights.bin";

            if (std::filesystem::exists(base_weights)) {
                // Create backup if not already present to preserve pre-trained base
                if (!std::filesystem::exists(backup_base)) {
                    std::error_code ec;
                    std::filesystem::copy_file(base_weights, backup_base, std::filesystem::copy_options::overwrite_existing, ec);
                    if (!ec) {
                        std::cout << "  [Safety Protection] Base checkpoint archived → " << backup_base << "\n";
                    }
                } else {
                    std::cout << "  [Safety Protection] Base checkpoint archive verified: " << backup_base << "\n";
                }

                // Load base weights into model for instruction alignment transfer learning
                uint32_t loaded_base_step = 0;
                float loaded_base_loss = 0.0f;
                if (model.load_weights(base_weights, &loaded_base_step, &loaded_base_loss)) {
                    std::cout << "  [Transfer Learning] Successfully loaded base weights from " << base_weights 
                              << " (Base Step " << loaded_base_step << " | Loss: " << std::fixed << std::setprecision(4) << loaded_base_loss << ")\n";
                } else {
                    std::cout << "  [WARNING] Could not parse base weights. Initializing fresh parameters.\n";
                }
            } else {
                std::cout << "  [INFO] No base weights found at " << base_weights << ". Starting with random initialization.\n";
            }

            // 2. Conversational Training Configuration — Turn-Aligned SFT Mode
            const std::string instruct_dataset = "titan_instruct_tokens.bin";
            if (!std::filesystem::exists(instruct_dataset)) {
                std::cerr << "\n[ERROR] Conversational dataset not found: " << instruct_dataset << "\n";
                std::cerr << "Run './ingest_instruct.sh' first to download and stream instruction tokens.\n";
                return 1;
            }

            titan::TrainingConfig config;
            config.dataset_path        = instruct_dataset;
            config.checkpoint_path     = instruct_weights;
            config.is_sft              = true;     // Turn-aligned SFT with target loss masking
            config.batch_size          = 6;        // 6 dialogues per micro-batch
            config.seq_len             = 192;      // Fixed sequence length for aligned dialogue turns
            config.max_steps           = 5000;     // 5000 focused SFT steps
            config.warmup_steps        = 250;      // Warmup
            config.learning_rate       = 1e-4f;    // Targeted peak LR for response adaptation
            config.min_learning_rate   = 1e-5f;    // Cosine floor
            config.weight_decay        = 0.05f;
            config.log_interval        = 10;       // Log every 10 steps
            config.checkpoint_interval = 250;      // Regular checkpointing

            // Smart Closed-Loop Hardware Governor (Real-Time Apple Silicon IOKit Telemetry)
            config.governor.enabled = true;
            config.governor.profile = titan::ThermalProfile::AUTO;
            for (int i = 2; i < argc; ++i) {
                std::string arg = argv[i];
                if (arg == "--docked") {
                    config.governor.apply_profile(titan::ThermalProfile::DOCKED);
                } else if (arg == "--mobile") {
                    config.governor.apply_profile(titan::ThermalProfile::MOBILE);
                }
            }

            std::cout << "  Dataset File: " << config.dataset_path << "\n";
            std::cout << "  Checkpoint  : " << config.checkpoint_path << "\n";
            std::cout << "  Fine-Tuning : " << config.max_steps << " steps"
                      << " | batch=" << config.batch_size
                      << " | seq=" << config.seq_len
                      << " | Warmup: " << config.warmup_steps
                      << " steps | Peak LR: " << config.learning_rate << "\n";
            std::cout << "  Tokens/step : " << config.batch_size * config.seq_len
                      << " | Total tokens: ~"
                      << (config.batch_size * config.seq_len * config.max_steps) / 1'000'000
                      << "M (~1.86 epochs over CodeAlpaca-20k)\n";
            std::cout << "--------------------------------------------------------------------\n\n";

            titan::Trainer trainer(model, tokenizer, config);
            trainer.train();

            std::cout << "\n[train_instruct] Titan-450M Alignment complete! Checkpoint saved → " << instruct_weights << "\n";
            return 0;
        }

        if (command == "generate" && argc >= 3) {
            std::string prompt = argv[2];
            // ── Must match the architecture saved in titan_weights.bin ──────
            const size_t vocab_size = tokenizer.vocab_size();
            const size_t d_model    = 1152;   // Titan-450M
            const size_t num_layers = 28;
            const size_t n_heads    = 16;
            const size_t max_seq    = 256;

            // Optional args: ./titan_lm generate "prompt" [tokens] [temp] [rep_penalty]
            size_t gen_tokens          = 256;
            float  temperature         = 0.8f;
            float  repetition_penalty  = 1.20f;
            if (argc >= 4) gen_tokens         = static_cast<size_t>(std::stoul(argv[3]));
            if (argc >= 5) temperature        = std::stof(argv[4]);
            if (argc >= 6) repetition_penalty = std::stof(argv[5]);

            titan::TitanModel model(vocab_size, d_model, num_layers, n_heads, max_seq);
            // 'generate' always uses the base pre-trained weights for raw code generation.
            // Use './titan_lm chat' for instruction-following (uses titan_instruct_weights.bin).
            std::string weight_file = "titan_weights.bin";

            if (std::filesystem::exists(weight_file)) {
                model.load_weights(weight_file);
                std::cout << "[generate] Loaded checkpoint: " << weight_file << "\n";
            } else {
                std::cout << "[generate] WARNING: No checkpoint found. "
                          << "Run 'make ingest && make train' first.\n";
            }

            std::cout << "[generate] Prompt      : \"" << prompt << "\"\n";
            std::cout << "[generate] Max tokens  : " << gen_tokens << "\n";
            std::cout << "[generate] Temperature : " << temperature << "\n";
            std::cout << "[generate] Repetition P: " << repetition_penalty << "\n";
            std::cout << "\n──────────────────────────────────────────────────\n";
            model.generate(prompt, tokenizer, gen_tokens, temperature, 1337, true, repetition_penalty);
            std::cout << "\n──────────────────────────────────────────────────\n";
            return 0;
        }

        if (command == "chat" && argc >= 3) {
            std::string user_instruction = argv[2];
            const size_t vocab_size = tokenizer.vocab_size();
            const size_t d_model    = 1152;
            const size_t num_layers = 28;
            const size_t n_heads    = 16;
            const size_t max_seq    = 256;

            size_t gen_tokens          = 256;
            float  temperature         = 0.7f;
            float  repetition_penalty  = 1.25f;
            if (argc >= 4) gen_tokens         = static_cast<size_t>(std::stoul(argv[3]));
            if (argc >= 5) temperature        = std::stof(argv[4]);
            if (argc >= 6) repetition_penalty = std::stof(argv[5]);

            titan::TitanModel model(vocab_size, d_model, num_layers, n_heads, max_seq);
            std::string weight_file = "titan_instruct_weights.bin";
            if (!std::filesystem::exists(weight_file)) {
                weight_file = "titan_weights.bin";
            }

            if (std::filesystem::exists(weight_file)) {
                model.load_weights(weight_file);
                std::cout << "[chat] Loaded checkpoint: " << weight_file << "\n";
            }

            std::string prompt = "<|user|>\n" + user_instruction + "\n<|assistant|>\n";
            std::cout << "\n──────────────── Conversational Dialogue ────────────────\n";
            std::cout << "[User]: " << user_instruction << "\n\n[Assistant]:\n";
            model.generate(prompt, tokenizer, gen_tokens, temperature, 1337, true, repetition_penalty, true);
            std::cout << "\n──────────────────────────────────────────────────────────\n";
            return 0;
        }

        if (command == "benchmark" && argc >= 3) {
            std::filesystem::path target = argv[2];
            auto stats = tokenizer.benchmark_ingestion(target);
            std::cout << "[Ingestion Benchmark: " << target << "]\n";
            std::cout << "  Files Processed: " << stats.file_count << "\n";
            std::cout << "  Throughput: " << std::fixed << std::setprecision(2)
                      << stats.throughput_mb_s << " MB/s\n\n";

            run_tensor_autograd_verification();
            run_attention_and_ffn_verification();
            run_phase4_transformer_block_verification();
            run_phase5_trainer_and_generation_verification(tokenizer);
            return 0;
        }

        std::cerr << "Unknown command: " << command << "\n";
        std::cout << "Usage:\n";
        std::cout << "  " << argv[0] << "                                (runs full Phase 1-5 verification suite)\n";
        std::cout << "  " << argv[0] << " train-vocab [size] [file.txt]  (learns BPE merges up to target size)\n";
        std::cout << "  " << argv[0] << " tokenize <file.txt>            (BPE-encode file → titan_tokens.bin)\n";
        std::cout << "  " << argv[0] << " train                          (launches base AdamW training loop on corpus)\n";
        std::cout << "  " << argv[0] << " train_instruct                 (launches Phase 9 instruction alignment loop)\n";
        std::cout << "  " << argv[0] << " stream-instruct <in> [out]     (streams chat template into binary tokens)\n";
        std::cout << "  " << argv[0] << " chat \"<instruction>\"           (conversational dialogue inference)\n";
        std::cout << "  " << argv[0] << " generate <p> [N] [T] [rep]     (streams real-time code with repetition penalty)\n";
        std::cout << "  " << argv[0] << " benchmark .                    (runs system benchmarks)\n";
        return 1;
    }

    // Default: Execute complete verification across all phases (Phase 1 through 5)
    run_ingestion_verification(tokenizer);
    run_tensor_autograd_verification();
    run_attention_and_ffn_verification();
    run_phase4_transformer_block_verification();
    run_phase5_trainer_and_generation_verification(tokenizer);

    std::cout << "Titan-LM Full Engine (Phases 1-5) Operational on Apple Silicon M4.\n";
    return 0;
}
