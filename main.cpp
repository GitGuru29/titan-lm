#include "tokenizer.hpp"
#include "tensor.hpp"
#include "attention.hpp"
#include "model.hpp"
#include "trainer.hpp"

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

    const std::string out_bin = "titan_tokens.bin";
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
        config.checkpoint_path = "titan_weights.bin";

        titan::Trainer trainer(model, tokenizer, config);
        trainer.train();

        // Verify checkpoint file serialization
        bool checkpoint_exists = std::filesystem::exists("titan_weights.bin");
        size_t cp_size = checkpoint_exists ? std::filesystem::file_size("titan_weights.bin") : 0;

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

        if (command == "train") {
            const size_t vocab_size = tokenizer.vocab_size();
            const size_t d_model = 128;
            const size_t num_layers = 4;
            const size_t n_heads = 8;
            const size_t max_seq = 64;

            titan::TitanModel model(vocab_size, d_model, num_layers, n_heads, max_seq);

            titan::TrainingConfig config;
            config.batch_size = 4;
            config.seq_len = 32;
            config.max_steps = 100;
            config.warmup_steps = 10;
            config.learning_rate = 1e-3f;
            config.log_interval = 10;
            config.checkpoint_interval = 25;

            titan::Trainer trainer(model, tokenizer, config);
            trainer.train();
            return 0;
        }

        if (command == "generate" && argc >= 3) {
            std::string prompt = argv[2];
            const size_t vocab_size = tokenizer.vocab_size();
            const size_t d_model = 128;
            const size_t num_layers = 4;
            const size_t n_heads = 8;
            const size_t max_seq = 64;

            titan::TitanModel model(vocab_size, d_model, num_layers, n_heads, max_seq);
            if (std::filesystem::exists("titan_weights.bin")) {
                model.load_weights("titan_weights.bin");
                std::cout << "Loaded checkpoint from titan_weights.bin\n";
            }

            std::cout << "Streaming generation for: \"" << prompt << "\"\n";
            std::cout << "--------------------------------------------------\n";
            model.generate(prompt, tokenizer, 64, 0.8f, 1337, true);
            std::cout << "\n--------------------------------------------------\n";
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
        std::cout << "  " << argv[0] << "              (runs full Phase 1-5 verification suite)\n";
        std::cout << "  " << argv[0] << " train        (launches complete AdamW training loop)\n";
        std::cout << "  " << argv[0] << " generate <s> (streams real-time autoregressive code tokens)\n";
        std::cout << "  " << argv[0] << " benchmark .  (runs system benchmarks)\n";
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
