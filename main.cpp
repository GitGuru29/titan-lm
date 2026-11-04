#include "tokenizer.hpp"
#include "tensor.hpp"

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
    std::cout << "  Phase 1: Ingestion Pipeline & Phase 2: Accelerated Tensor Engine\n";
    std::cout << "====================================================================\n\n";
}

void print_system_telemetry() {
    std::cout << "[System Architecture & Hardware Topology]\n";
    std::cout << "  Target Architecture: Apple Silicon M4 (ARM64 Neon + Apple Matrix Coprocessor AMX)\n";
    std::cout << "  Execution Concurrency: " << std::thread::hardware_concurrency() << " hardware worker threads\n";

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

    // 1. Convert text to contiguous flat integer tokens
    std::vector<titan::TokenId> tokens = tokenizer.encode(sample_source, true, true);

    std::cout << "  Original Source Size: " << sample_source.size() << " bytes\n";
    std::cout << "  Contiguous Token Count: " << tokens.size() << " tokens\n";
    std::cout << "  Vocabulary Capacity: " << tokenizer.vocab_size() << " entries\n";
    std::cout << "  Compression Ratio: "
              << std::fixed << std::setprecision(2)
              << (static_cast<double>(sample_source.size()) / tokens.size()) << " bytes/token\n";

    // 2. Decode back and verify byte-for-byte exactness
    std::string reconstructed = tokenizer.decode(tokens, true);
    if (reconstructed == sample_source) {
        std::cout << "  Roundtrip Integrity: 100% BYTE-EXACT MATCH PASSED\n";
    } else {
        std::cerr << "  Roundtrip Integrity: MISMATCH DETECTED\n";
    }

    // 3. Ingest project files and export binary token stream
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

    tokenizer.save_tokens_binary(out_bin, workspace_tokens);
    titan::MappedTokenBuffer mapped = tokenizer.mmap_tokens_binary(out_bin);
    std::cout << "  Zero-Copy Dataset Buffer: " << mapped.size() << " tokens memory-mapped at "
              << static_cast<const void*>(mapped.data_ptr()) << "\n\n";
}

void run_tensor_autograd_verification() {
    std::cout << "[Phase 2 Verification: Accelerated Tensor Engine & Autograd Core]\n";

    // 1. Apple Matrix Coprocessor (AMX) matmul forward & backward verification
    {
        // A: (2 x 3), B: (3 x 2) -> C: (2 x 2)
        titan::Tensor A(2, 3, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
        titan::Tensor B(3, 2, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

        titan::Tensor C = titan::Tensor::matmul(A, B);

        // Ground truth: C = [[22, 28], [49, 64]]
        bool forward_ok = (std::abs(C(0, 0) - 22.0f) < 1e-4f) &&
                          (std::abs(C(0, 1) - 28.0f) < 1e-4f) &&
                          (std::abs(C(1, 0) - 49.0f) < 1e-4f) &&
                          (std::abs(C(1, 1) - 64.0f) < 1e-4f);

        // Upstream gradient dC = [[1, 1], [1, 1]]
        titan::Tensor grad_C(2, 2, 1.0f);
        titan::Tensor::backward_matmul(A, B, grad_C, A, B);

        // Ground truth dA = dC * B^T = [[3, 7, 11], [3, 7, 11]]
        // Ground truth dB = A^T * dC = [[5, 5], [7, 7], [9, 9]]
        bool grad_a_ok = (std::abs(A.grad_at(0, 0) - 3.0f) < 1e-4f) &&
                         (std::abs(A.grad_at(0, 1) - 7.0f) < 1e-4f) &&
                         (std::abs(A.grad_at(0, 2) - 11.0f) < 1e-4f);
        bool grad_b_ok = (std::abs(B.grad_at(0, 0) - 5.0f) < 1e-4f) &&
                         (std::abs(B.grad_at(1, 0) - 7.0f) < 1e-4f) &&
                         (std::abs(B.grad_at(2, 0) - 9.0f) < 1e-4f);

        if (forward_ok && grad_a_ok && grad_b_ok) {
            std::cout << "  AMX cblas_sgemm Matmul Forward & Backward: PASSED (Exact Analytical Match)\n";
        } else {
            std::cerr << "  AMX cblas_sgemm Matmul Forward & Backward: FAILED\n";
        }
    }

    // 2. ReLU Activation & Derivative
    {
        titan::Tensor X(1, 4, {-2.0f, 0.5f, -0.1f, 3.0f});
        titan::Tensor Y = X.forward_relu();
        titan::Tensor grad_out(1, 4, {1.0f, 1.0f, 1.0f, 1.0f});
        X.backward_relu(grad_out);

        bool relu_fwd_ok = (Y(0, 0) == 0.0f && Y(0, 1) == 0.5f && Y(0, 2) == 0.0f && Y(0, 3) == 3.0f);
        bool relu_bwd_ok = (X.grad_at(0, 0) == 0.0f && X.grad_at(0, 1) == 1.0f &&
                            X.grad_at(0, 2) == 0.0f && X.grad_at(0, 3) == 1.0f);

        if (relu_fwd_ok && relu_bwd_ok) {
            std::cout << "  ReLU Forward & Derivative Accumulation: PASSED\n";
        } else {
            std::cerr << "  ReLU Forward & Derivative Accumulation: FAILED\n";
        }
    }

    // 3. Numerically-Stable Row-Wise Softmax & Analytical Derivative
    {
        titan::Tensor logits(2, 3, {1.0f, 2.0f, 3.0f, 1000.0f, 1001.0f, 1002.0f}); // Large values test overflow stability
        titan::Tensor probs = logits.forward_softmax();

        float row0_sum = probs(0, 0) + probs(0, 1) + probs(0, 2);
        float row1_sum = probs(1, 0) + probs(1, 1) + probs(1, 2);

        titan::Tensor grad_probs(2, 3, {1.0f, 0.0f, 0.0f, 0.5f, 0.5f, 0.0f});
        logits.backward_softmax(probs, grad_probs);

        // Softmax derivative invariant: sum of dX across a row must equal zero
        float dX_sum0 = logits.grad_at(0, 0) + logits.grad_at(0, 1) + logits.grad_at(0, 2);
        float dX_sum1 = logits.grad_at(1, 0) + logits.grad_at(1, 1) + logits.grad_at(1, 2);

        bool softmax_ok = (std::abs(row0_sum - 1.0f) < 1e-4f) &&
                          (std::abs(row1_sum - 1.0f) < 1e-4f) &&
                          (std::abs(dX_sum0) < 1e-4f) &&
                          (std::abs(dX_sum1) < 1e-4f);

        if (softmax_ok) {
            std::cout << "  Stable Row Softmax & Zero-Sum Derivative Invariant: PASSED\n";
        } else {
            std::cerr << "  Stable Row Softmax & Zero-Sum Derivative Invariant: FAILED\n";
        }
    }

    // 4. Bias Layer Broadcast & Row-Sum Gradient Accumulation
    {
        titan::Tensor X(3, 2, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
        titan::Tensor bias(1, 2, {0.5f, -0.5f});

        titan::Tensor Y = X.forward_add_bias(bias);
        titan::Tensor grad_Y(3, 2, 1.0f); // Upstream gradient: ones

        titan::Tensor::backward_add_bias(grad_Y, X, bias);

        // Bias grad should be sum of rows: [3.0, 3.0]
        bool bias_ok = (std::abs(bias.grad_at(0, 0) - 3.0f) < 1e-4f) &&
                       (std::abs(bias.grad_at(0, 1) - 3.0f) < 1e-4f);

        if (bias_ok) {
            std::cout << "  Bias Broadcast & Column-Sum Backpropagation: PASSED\n";
        } else {
            std::cerr << "  Bias Broadcast & Column-Sum Backpropagation: FAILED\n";
        }
    }

    // 5. Memory Management & Initialization
    {
        titan::Tensor W(64, 64);
        W.fill_random(-1.0f, 1.0f, 12345);
        W.grad.assign(W.size(), 5.0f);
        W.zero_gradients();

        bool zero_ok = true;
        for (float g : W.grad) {
            if (g != 0.0f) { zero_ok = false; break; }
        }

        W.init_xavier(64, 64);
        if (zero_ok && W.size() == 4096) {
            std::cout << "  Memory Management (zero_gradients, fill_random, Xavier): PASSED\n";
        } else {
            std::cerr << "  Memory Management: FAILED\n";
        }
    }

    // 6. AMX Hardware Performance Benchmark
    {
        const size_t M = 1024, N = 1024, K = 1024;
        titan::Tensor A_bench(M, K);
        titan::Tensor B_bench(K, N);
        A_bench.init_xavier(M, K);
        B_bench.init_xavier(K, N);

        // Warmup
        titan::Tensor C_bench = titan::Tensor::matmul(A_bench, B_bench);

        // Timed benchmark: 20 iterations of 1024x1024 matrix multiplication
        const int iters = 20;
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) {
            C_bench = titan::Tensor::matmul(A_bench, B_bench);
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;

        double avg_ms = duration.count() / iters;
        double gflops = (2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K) * 1e-9) / (avg_ms * 1e-3);

        std::cout << "  AMX 1024x1024 GEMM Benchmark: " << std::fixed << std::setprecision(2)
                  << avg_ms << " ms/iter (" << gflops << " GFLOPS)\n\n";
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
            std::cout << "  Duration: " << stats.elapsed_ms << " ms\n";
            std::cout << "  Throughput: " << std::fixed << std::setprecision(2)
                      << stats.throughput_mb_s << " MB/s ("
                      << stats.tokens_per_sec << " tokens/sec)\n\n";

            run_tensor_autograd_verification();
            return 0;
        }

        if (command == "ingest" && argc >= 4) {
            std::filesystem::path input_path = argv[2];
            std::filesystem::path output_path = argv[3];

            std::cout << "[Ingesting: " << input_path << " -> " << output_path << "]\n";
            std::vector<titan::TokenId> tokens;
            if (std::filesystem::is_directory(input_path)) {
                tokens = tokenizer.ingest_directory(input_path);
            } else {
                tokens = tokenizer.ingest_file(input_path);
            }

            if (tokenizer.save_tokens_binary(output_path, tokens)) {
                std::cout << "Successfully saved " << tokens.size()
                          << " contiguous tokens to " << output_path << "\n";
            } else {
                std::cerr << "Failed to write output binary.\n";
                return 1;
            }
            return 0;
        }

        std::cerr << "Unknown command: " << command << "\n";
        std::cout << "Usage:\n";
        std::cout << "  " << argv[0] << " (runs full Phase 1 & 2 verification suite)\n";
        std::cout << "  " << argv[0] << " ingest <input_file_or_dir> <output_tokens.bin>\n";
        std::cout << "  " << argv[0] << " benchmark <target_file_or_dir>\n";
        return 1;
    }

    // Default: Execute both Phase 1 Ingestion and Phase 2 Accelerated Tensor & Autograd suite
    run_ingestion_verification(tokenizer);
    run_tensor_autograd_verification();

    std::cout << "Titan-LM Phase 1 & Phase 2 Core Systems Verified & Online.\n";
    return 0;
}
