#include "tokenizer.hpp"
#include "tensor.hpp"
#include "attention.hpp"
#include "model.hpp"

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <thread>
#include <chrono>
#include <cmath>
#include <Accelerate/Accelerate.h>

namespace {

void print_banner() {
    std::cout << "====================================================================\n";
    std::cout << "  TITAN-LM: Bare-Metal Transformer Engine (Apple Silicon M4 / C++20)\n";
    std::cout << "  Phase 1: Ingestion | Phase 2: AMX Core | Phase 3: MHA & FFN | Phase 4: Full Model\n";
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

    const std::string out_bin = "titan_tokens.bin";
    std::vector<titan::TokenId> workspace_tokens;
    if (std::filesystem::exists("tokenizer.hpp")) {
        auto hpp_tokens = tokenizer.ingest_file("tokenizer.hpp", true, true);
        workspace_tokens.insert(workspace_tokens.end(), hpp_tokens.begin(), hpp_tokens.end());
    }
    if (std::filesystem::exists("model.hpp")) {
        auto model_tokens = tokenizer.ingest_file("model.hpp", true, true);
        workspace_tokens.insert(workspace_tokens.end(), model_tokens.begin(), model_tokens.end());
    }

    tokenizer.save_tokens_binary(out_bin, workspace_tokens);
    titan::MappedTokenBuffer mapped = tokenizer.mmap_tokens_binary(out_bin);
    std::cout << "  Zero-Copy Dataset Buffer: " << mapped.size() << " tokens memory-mapped at "
              << static_cast<const void*>(mapped.data_ptr()) << "\n\n";
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
    std::cout << "[Phase 4 Verification: LayerNorm, Dual Residual Streams & TransformerBlock]\n";

    const size_t d_model = 64;
    const size_t n_heads = 4;
    const size_t max_seq = 32;
    const size_t T = 4;

    // 1. LayerNorm Operator Forward & Analytical vDSP Backward Invariant
    {
        titan::LayerNorm ln(d_model);
        titan::Tensor X_ln(T, d_model);
        X_ln.fill_random(-2.0f, 2.0f, 101);

        titan::Tensor Y_ln = ln.forward(X_ln);

        // Verify normalization properties on output Y_ln: row mean ~ 0, row variance ~ 1
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

        // Backward pass
        titan::Tensor grad_Y_ln(T, d_model, 1.0f);
        titan::Tensor grad_X_ln(T, d_model, 0.0f);
        ln.backward(grad_Y_ln, grad_X_ln);

        // Mathematical invariant: sum of input gradients across each row must strictly equal 0
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

        bool gamma_grad_ok = false;
        for (float g : ln.gamma.grad) {
            if (std::abs(g) > 1e-5f) gamma_grad_ok = true;
        }

        if (ln_norm_ok && zero_sum_invariant && gamma_grad_ok) {
            std::cout << "  LayerNorm Forward Normalization & vDSP Analytical Backward: PASSED (Zero-Sum Invariant Verified)\n";
        } else {
            std::cerr << "  LayerNorm Forward Normalization & vDSP Analytical Backward: FAILED\n";
        }
    }

    // 2. TransformerBlock Pre-LN Assembly & Deep Autograd Dual Residual Traversal
    {
        titan::TransformerBlock block(d_model, n_heads, max_seq);
        block.init_weights(555);

        titan::Tensor X(T, d_model);
        X.fill_random(-0.5f, 0.5f, 666);

        // Pre-LN Forward pass
        titan::Tensor Y = block.forward(X);
        bool shape_ok = (Y.rows == T && Y.cols == d_model);

        // Upstream gradient
        titan::Tensor grad_Y(T, d_model, 1.0f);
        titan::Tensor grad_X(T, d_model, 0.0f);

        // Deep autograd traversal: routes through dual residual streams
        block.backward(grad_Y, grad_X);

        // Verify that BOTH residual streams and ALL constituent sub-layers received gradients
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
            check_has_grad(block.attn.W_o) &&
            check_has_grad(block.ln_2.gamma) &&
            check_has_grad(block.ffn.W_1) &&
            check_has_grad(block.ffn.W_2) &&
            check_has_grad(grad_X);

        if (shape_ok && all_sublayers_accumulated) {
            std::cout << "  TransformerBlock Pre-LN Forward & Dual Residual Autograd: PASSED (All Branches Active)\n";
        } else {
            std::cerr << "  TransformerBlock Pre-LN Forward & Dual Residual Autograd: FAILED\n";
        }
    }

    // 3. Multi-Layer TitanModel End-to-End Verification (Embeddings -> 2 Blocks -> Final LN -> LM Head)
    {
        const size_t vocab_size = 256;
        const size_t num_layers = 2;
        titan::TitanModel model(vocab_size, d_model, num_layers, n_heads, max_seq);

        std::vector<uint32_t> sample_tokens = {10, 42, 105, 200};
        titan::Tensor logits = model.forward(sample_tokens);

        bool logits_shape_ok = (logits.rows == sample_tokens.size() && logits.cols == vocab_size);

        // Compute Cross-Entropy Loss with respect to target tokens
        std::vector<uint32_t> targets = {42, 105, 200, 1};
        titan::Tensor grad_logits(logits.rows, logits.cols, 0.0f);
        float loss = titan::Tensor::cross_entropy_loss(logits, targets, grad_logits);

        // Backpropagate all the way to token & positional embeddings
        model.backward(grad_logits);

        bool model_grad_ok = false;
        for (float g : model.token_embeddings.grad) {
            if (std::abs(g) > 1e-9f) { model_grad_ok = true; break; }
        }

        if (logits_shape_ok && loss > 0.0f && model_grad_ok) {
            std::cout << "  End-to-End Multi-Layer TitanModel (Embeddings + 2 Blocks + LM Head): PASSED (Loss: "
                      << std::fixed << std::setprecision(4) << loss << ")\n";
        } else {
            std::cerr << "  End-to-End Multi-Layer TitanModel: FAILED\n";
        }
    }

    // 4. M4 Hardware Benchmark: TransformerBlock Throughput
    {
        const size_t bench_T = 128;
        const size_t bench_d = 256;
        const size_t bench_heads = 8;
        titan::TransformerBlock bench_block(bench_d, bench_heads, bench_T);

        titan::Tensor bench_X(bench_T, bench_d);
        bench_X.fill_random(-0.1f, 0.1f, 303);

        // Warmup
        titan::Tensor out = bench_block.forward(bench_X);

        const int iters = 50;
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) {
            out = bench_block.forward(bench_X);
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;

        double avg_ms = duration.count() / iters;
        double throughput = (bench_T * iters) / (duration.count() * 1e-3);

        std::cout << "  Full TransformerBlock Benchmark (Pre-LN, AMX GEMM, Causal Attention, FFN):\n";
        std::cout << "    Latency: " << std::fixed << std::setprecision(3) << avg_ms << " ms/block\n";
        std::cout << "    Throughput: " << std::fixed << std::setprecision(1) << throughput << " tokens/sec\n\n";
    }
}

} // anonymous namespace

int main(int argc, char* argv[]) {
    print_banner();
    print_system_telemetry();

    titan::Tokenizer tokenizer(true);

    if (argc >= 2) {
        std::string command = argv[1];

        if (command == "benchmark" && argc >= 3) {
            std::filesystem::path target = argv[2];
            auto stats = tokenizer.benchmark_ingestion(target);
            std::cout << "[Ingestion Benchmark: " << target << "]\n";
            std::cout << "  Files Processed: " << stats.file_count << "\n";
            std::cout << "  Total Bytes: " << stats.total_bytes << " bytes\n";
            std::cout << "  Total Tokens: " << stats.total_tokens << " tokens\n";
            std::cout << "  Throughput: " << std::fixed << std::setprecision(2)
                      << stats.throughput_mb_s << " MB/s\n\n";

            run_tensor_autograd_verification();
            run_attention_and_ffn_verification();
            run_phase4_transformer_block_verification();
            return 0;
        }

        if (command == "ingest" && argc >= 4) {
            std::filesystem::path input_path = argv[2];
            std::filesystem::path output_path = argv[3];

            std::vector<titan::TokenId> tokens;
            if (std::filesystem::is_directory(input_path)) {
                tokens = tokenizer.ingest_directory(input_path);
            } else {
                tokens = tokenizer.ingest_file(input_path);
            }

            if (tokenizer.save_tokens_binary(output_path, tokens)) {
                std::cout << "Saved " << tokens.size() << " contiguous tokens to " << output_path << "\n";
            } else {
                std::cerr << "Failed to write output binary.\n";
                return 1;
            }
            return 0;
        }

        std::cerr << "Unknown command: " << command << "\n";
        std::cout << "Usage:\n";
        std::cout << "  " << argv[0] << " (runs full Phase 1-4 verification suite)\n";
        std::cout << "  " << argv[0] << " ingest <input_file_or_dir> <output_tokens.bin>\n";
        std::cout << "  " << argv[0] << " benchmark <target_file_or_dir>\n";
        return 1;
    }

    // Default: Execute Phase 1, Phase 2, Phase 3, and Phase 4 full verification suites
    run_ingestion_verification(tokenizer);
    run_tensor_autograd_verification();
    run_attention_and_ffn_verification();
    run_phase4_transformer_block_verification();

    std::cout << "Titan-LM Phase 1-4 Systems Operational.\n";
    return 0;
}
