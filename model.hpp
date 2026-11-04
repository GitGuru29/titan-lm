#pragma once
#ifndef TITAN_MODEL_HPP
#define TITAN_MODEL_HPP

#include "tensor.hpp"
#include "attention.hpp"
#include <vector>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace titan {

// Forward declarations
class Tokenizer;

// Parameter reference with weight decay flag for AdamW
struct ParameterRef {
    Tensor* tensor{nullptr};
    std::string name;
    bool weight_decay{true};
};

// ============================================================================
// Layer Normalization Cache (Autograd Intermediate Buffer)
// ============================================================================
struct LayerNormCache {
    Tensor X;                // Input tensor (T x d_model)
    Tensor X_hat;            // Zero-mean unit-variance normalized tensor (T x d_model)
    std::vector<float> mean; // Row-wise means (size T)
    std::vector<float> rstd; // Row-wise reciprocal standard deviation 1/sqrt(var + eps) (size T)
};

// ============================================================================
// Layer Normalization Operator with Learnable Gain and Bias (Pre-LN Standard)
// ============================================================================
class LayerNorm {
public:
    size_t d_model{0};
    float eps{1e-5f};

    // Learnable parameters
    Tensor gamma; // Gain vector (1 x d_model), initialized to 1.0f
    Tensor beta;  // Bias vector (1 x d_model), initialized to 0.0f

    // Autograd forward cache
    LayerNormCache cache;

    // --- Lifecycle ---
    LayerNorm() = default;
    explicit LayerNorm(size_t d_model, float eps = 1e-5f);

    ~LayerNorm() = default;
    LayerNorm(const LayerNorm&) = default;
    LayerNorm& operator=(const LayerNorm&) = default;
    LayerNorm(LayerNorm&&) noexcept = default;
    LayerNorm& operator=(LayerNorm&&) noexcept = default;

    // --- Forward & Backward Core ---

    // Forward pass: calculates row-wise mean and variance, normalizes,
    // and applies gain (gamma) and bias (beta) using vectorized Apple Accelerate vDSP
    [[nodiscard]] Tensor forward(const Tensor& X);

    // Backward pass: calculates exact analytical derivatives for gamma, beta,
    // and the input tensor without loops by leveraging Apple Accelerate vDSP
    void backward(const Tensor& grad_output, Tensor& grad_X);

    // --- Memory Management ---
    void init_weights();
    void zero_gradients() noexcept;
    void clear_cache() noexcept;

    [[nodiscard]] size_t parameter_count() const noexcept {
        return gamma.size() + beta.size();
    }
};

// ============================================================================
// Transformer Block Autograd Cache
// ============================================================================
struct TransformerBlockCache {
    Tensor X;         // Block input (T x d_model)
    Tensor X_norm1;   // Output of LN1 (T x d_model)
    Tensor X_attn;    // Output of Causal Attention (T x d_model)
    Tensor X_mid;     // Intermediate residual: X + X_attn (T x d_model)
    Tensor X_norm2;   // Output of LN2 (T x d_model)
    Tensor X_ffn;     // Output of FeedForward (T x d_model)
};

// ============================================================================
// Full Transformer Block Assembly (Pre-LN Architecture with Dual Residual Streams)
// ============================================================================
class TransformerBlock {
public:
    size_t d_model{0};
    size_t n_heads{0};
    size_t d_ff{0};
    size_t max_seq_len{0};

    // Sub-layer operators
    LayerNorm ln_1;          // Pre-Attention LayerNorm
    CausalAttention attn;    // Causal Multi-Head Self-Attention
    LayerNorm ln_2;          // Pre-FFN LayerNorm
    FeedForward ffn;         // Two-stage MLP with ReLU activation

    // Autograd cache
    TransformerBlockCache cache;

    // --- Lifecycle ---
    TransformerBlock() = default;
    TransformerBlock(
        size_t d_model,
        size_t n_heads,
        size_t max_seq_len,
        size_t d_ff = 0,
        float eps = 1e-5f
    );

    ~TransformerBlock() = default;
    TransformerBlock(const TransformerBlock&) = default;
    TransformerBlock& operator=(const TransformerBlock&) = default;
    TransformerBlock(TransformerBlock&&) noexcept = default;
    TransformerBlock& operator=(TransformerBlock&&) noexcept = default;

    // --- Forward & Backward Core ---

    // Pre-LN Forward Pass:
    // 1. X_norm1 = ln_1(X)
    // 2. X_attn  = attn(X_norm1)
    // 3. X_mid   = X + X_attn           (First Residual Stream)
    // 4. X_norm2 = ln_2(X_mid)
    // 5. X_ffn   = ffn(X_norm2)
    // 6. Y       = X_mid + X_ffn        (Second Residual Stream)
    [[nodiscard]] Tensor forward(const Tensor& X);

    // Deep Autograd Traversal Backward Pass:
    // Routes gradients down through the dual residual connections and LayerNorm operators,
    // accumulating parameter gradients across all sub-layers without dropping branches.
    void backward(const Tensor& grad_output, Tensor& grad_X);

    // --- Memory Management & Parameter Inspection ---
    void init_weights(uint64_t seed = 42);
    void zero_gradients() noexcept;
    void clear_cache() noexcept;

    [[nodiscard]] size_t parameter_count() const noexcept {
        return ln_1.parameter_count() + attn.parameter_count() +
               ln_2.parameter_count() + ffn.parameter_count();
    }
};

// ============================================================================
// Multi-Layer Transformer Architecture (Stack of TransformerBlocks)
// ============================================================================
class TitanModel {
public:
    size_t vocab_size{0};
    size_t d_model{0};
    size_t num_layers{0};
    size_t n_heads{0};
    size_t max_seq_len{0};

    // Embedding parameters
    Tensor token_embeddings; // (vocab_size x d_model)
    Tensor pos_embeddings;   // (max_seq_len x d_model)

    // Stack of Transformer Blocks
    std::vector<TransformerBlock> layers;

    // Final Layer Normalization
    LayerNorm final_ln;

    // Language Model Head Projection (d_model x vocab_size)
    Tensor lm_head;

    // Forward Cache for multi-layer backpropagation
    std::vector<Tensor> layer_inputs;
    Tensor cached_final_norm;
    std::vector<uint32_t> cached_input_tokens;

    // --- Lifecycle ---
    TitanModel() = default;
    TitanModel(
        size_t vocab_size,
        size_t d_model,
        size_t num_layers,
        size_t n_heads,
        size_t max_seq_len,
        size_t d_ff = 0
    );

    // Forward pass taking integer token sequence -> logits (T x vocab_size)
    [[nodiscard]] Tensor forward(const std::vector<uint32_t>& tokens);

    // Backward pass taking upstream grad_logits -> backpropagates through LM head, blocks & embeddings
    void backward(const Tensor& grad_logits);

    void init_weights(uint64_t seed = 42);
    void zero_gradients() noexcept;
    void clear_cache() noexcept;

    [[nodiscard]] size_t parameter_count() const noexcept;
    [[nodiscard]] std::vector<ParameterRef> get_parameters();

    // Model checkpoint serialization
    bool save_weights(const std::string& filepath, uint32_t step = 0, float loss = 0.0f) const;
    bool load_weights(const std::string& filepath);

    // Autoregressive token generation with real-time stdout streaming and repetition penalty
    std::string generate(
        const std::string& prompt,
        const Tokenizer& tokenizer,
        size_t max_new_tokens = 64,
        float temperature = 0.8f,
        uint64_t seed = 42,
        bool stream_to_stdout = true,
        float repetition_penalty = 1.20f
    );
};

} // namespace titan

#endif // TITAN_MODEL_HPP

