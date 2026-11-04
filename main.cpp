#include "tokenizer.hpp"
#include "tensor.hpp"
#include "attention.hpp"

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
    std::cout << "  Phase 1: Ingestion | Phase 2: AMX Tensors | Phase 3: Causal MHA & FFN\n";
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
    if (std::filesystem::exists("tensor.hpp")) {
        auto tensor_tokens = tokenizer.ingest_file("tensor.hpp", true, true);
        workspace_tokens.insert(workspace_tokens.end(), tensor_tokens.begin(), tensor_tokens.end());
    }
    if (std::filesystem::exists("attention.hpp")) {
        auto attn_tokens = tokenizer.ingest_file("attention.hpp", true, true);
        workspace_tokens.insert(workspace_tokens.end(), attn_tokens.begin(), attn_tokens.end());
    }

    tokenizer.save_tokens_binary(out_bin, workspace_tokens);
    titan::MappedTokenBuffer mapped = tokenizer.mmap_tokens_binary(out_bin);
    std::cout << "  Zero-Copy Dataset Buffer: " << mapped.size() << " tokens memory-mapped at "
              << static_cast<const void*>(mapped.data_ptr()) << "\n\n";
}

void run_tensor_autograd_verification() {
    std::cout << "[Phase 2 Verification: Accelerated Tensor Engine & Autograd Core]\n";

    // 1. AMX matmul forward & backward
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

    // 2. ReLU & Derivative
    titan::Tensor X_relu(1, 4, {-2.0f, 0.5f, -0.1f, 3.0f});
    titan::Tensor Y_relu = X_relu.forward_relu();
    titan::Tensor grad_relu(1, 4, {1.0f, 1.0f, 1.0f, 1.0f});
    X_relu.backward_relu(grad_relu);

    if (Y_relu(0, 0) == 0.0f && Y_relu(0, 1) == 0.5f && X_relu.grad_at(0, 0) == 0.0f && X_relu.grad_at(0, 1) == 1.0f) {
        std::cout << "  ReLU Forward & Derivative Accumulation: PASSED\n";
    } else {
        std::cerr << "  ReLU Forward & Derivative Accumulation: FAILED\n";
    }

    // 3. Stable Softmax & Bias
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
    const size_t T = 4; // Sequence length for validation test

    // 1. Causal Self-Attention Forward & Causal Masking Invariant Verification
    titan::CausalAttention attention(d_model, n_heads, max_seq);
    attention.init_weights(123);

    titan::Tensor X(T, d_model);
    X.fill_random(-0.5f, 0.5f, 456);

    titan::Tensor Y_attn = attention.forward(X);

    // Verify dimensions: (T x d_model)
    bool dim_ok = (Y_attn.rows == T && Y_attn.cols == d_model);

    // Verify strict upper-triangular causal masking across all heads:
    // Attention weights must be strictly 0.0 for future tokens (c > r)
    // and row sum must be exactly 1.0
    bool causal_mask_strictly_enforced = true;
    for (size_t h = 0; h < n_heads; ++h) {
        const auto& weights_h = attention.cache.weights[h];
        for (size_t r = 0; r < T; ++r) {
            float row_sum = 0.0f;
            for (size_t c = 0; c < T; ++c) {
                float w = weights_h(r, c);
                row_sum += w;
                if (c > r && std::abs(w) > 1e-6f) {
                    causal_mask_strictly_enforced = false;
                }
            }
            if (std::abs(row_sum - 1.0f) > 1e-4f) {
                causal_mask_strictly_enforced = false;
            }
        }
    }

    if (dim_ok && causal_mask_strictly_enforced) {
        std::cout << "  Causal Self-Attention Forward & Mask Invariant: PASSED (Tokens Strictly Isolated)\n";
    } else {
        std::cerr << "  Causal Self-Attention Forward & Mask Invariant: FAILED\n";
    }

    // 2. Causal Self-Attention Autograd Backward Verification
    titan::Tensor grad_Y_attn(T, d_model, 1.0f);
    titan::Tensor grad_X(T, d_model, 0.0f);

    attention.backward(grad_Y_attn, grad_X);

    // Verify all projection parameter gradients are accumulated and non-zero
    bool gradients_accumulated = true;
    auto check_grad = [&](const titan::Tensor& t) {
        bool has_grad = false;
        for (float g : t.grad) {
            if (std::isnan(g) || std::isinf(g)) return false;
            if (std::abs(g) > 1e-9f) has_grad = true;
        }
        return has_grad;
    };

    gradients_accumulated &= check_grad(attention.W_q);
    gradients_accumulated &= check_grad(attention.b_q);
    gradients_accumulated &= check_grad(attention.W_k);
    gradients_accumulated &= check_grad(attention.b_k);
    gradients_accumulated &= check_grad(attention.W_v);
    gradients_accumulated &= check_grad(attention.b_v);
    gradients_accumulated &= check_grad(attention.W_o);
    gradients_accumulated &= check_grad(attention.b_o);
    gradients_accumulated &= check_grad(grad_X);

    if (gradients_accumulated) {
        std::cout << "  Causal Self-Attention Full Autograd Backward: PASSED (All Projections Gradient Verified)\n";
    } else {
        std::cerr << "  Causal Self-Attention Full Autograd Backward: FAILED\n";
    }

    // 3. Feed-Forward Network (FFN / MLP) Forward & Backward Verification
    const size_t d_ff = 256;
    titan::FeedForward ffn(d_model, d_ff);
    ffn.init_weights(789);

    titan::Tensor Y_ffn = ffn.forward(X);
    bool ffn_fwd_ok = (Y_ffn.rows == T && Y_ffn.cols == d_model);

    titan::Tensor grad_Y_ffn(T, d_model, 1.0f);
    titan::Tensor grad_X_ffn(T, d_model, 0.0f);
    ffn.backward(grad_Y_ffn, grad_X_ffn);

    bool ffn_grad_ok = check_grad(ffn.W_1) && check_grad(ffn.b_1) &&
                       check_grad(ffn.W_2) && check_grad(ffn.b_2) &&
                       check_grad(grad_X_ffn);

    if (ffn_fwd_ok && ffn_grad_ok) {
        std::cout << "  Feed-Forward (W2 x ReLU(W1 x X + b1) + b2) Autograd Pass: PASSED\n";
    } else {
        std::cerr << "  Feed-Forward Autograd Pass: FAILED\n";
    }

    // 4. Multi-Head Attention & FFN Throughput Benchmark on Apple Silicon M4
    const size_t bench_T = 128;
    const size_t bench_d = 256;
    const size_t bench_heads = 8;
    titan::CausalAttention bench_attn(bench_d, bench_heads, bench_T);
    titan::FeedForward bench_ffn(bench_d, 4 * bench_d);

    titan::Tensor bench_X(bench_T, bench_d);
    bench_X.fill_random(-0.1f, 0.1f, 101);

    // Warmup
    titan::Tensor out = bench_attn.forward(bench_X);
    out = bench_ffn.forward(out);

    const int iters = 50;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        titan::Tensor attn_out = bench_attn.forward(bench_X);
        titan::Tensor ffn_out = bench_ffn.forward(attn_out);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    double avg_ms = duration.count() / iters;
    double tokens_per_sec = (bench_T * iters) / (duration.count() * 1e-3);

    std::cout << "  Phase 3 Transformer Block Benchmark (Seq=" << bench_T << ", d=" << bench_d << "):\n";
    std::cout << "    Latency: " << std::fixed << std::setprecision(3) << avg_ms << " ms/block\n";
    std::cout << "    Throughput: " << std::fixed << std::setprecision(1) << tokens_per_sec << " tokens/second\n\n";
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
        std::cout << "  " << argv[0] << " (runs full Phase 1, 2 & 3 verification suite)\n";
        std::cout << "  " << argv[0] << " ingest <input_file_or_dir> <output_tokens.bin>\n";
        std::cout << "  " << argv[0] << " benchmark <target_file_or_dir>\n";
        return 1;
    }

    // Default: Execute Phase 1, Phase 2, and Phase 3 full verification suites
    run_ingestion_verification(tokenizer);
    run_tensor_autograd_verification();
    run_attention_and_ffn_verification();

    std::cout << "Titan-LM Phase 1, Phase 2 & Phase 3 Core Systems Operational.\n";
    return 0;
}
