#pragma once
#ifndef TITAN_ATTENTION_HPP
#define TITAN_ATTENTION_HPP

#include "tensor.hpp"
#include <vector>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace titan {

// ============================================================================
// Causal Multi-Head Attention Context (Backpropagation Cache)
// ============================================================================
struct AttentionCache {
    Tensor X;                   // Input activations (T x d_model)
    Tensor Q;                   // Query projection (T x d_model)
    Tensor K;                   // Key projection (T x d_model)
    Tensor V;                   // Value projection (T x d_model)
    std::vector<Tensor> Q_heads;// Sliced query heads (n_heads of T x d_k)
    std::vector<Tensor> K_heads;// Sliced key heads (n_heads of T x d_k)
    std::vector<Tensor> V_heads;// Sliced value heads (n_heads of T x d_k)
    std::vector<Tensor> scores; // Pre-softmax scaled causal scores (n_heads of T x T)
    std::vector<Tensor> weights;// Post-softmax causal attention probabilities (n_heads of T x T)
    std::vector<Tensor> head_outputs; // Head outputs (n_heads of T x d_k)
    Tensor concat_heads;        // Concatenated heads (T x d_model)
};

// ============================================================================
// Autoregressive Key-Value Cache (O(1) Step Decoding)
// Head-Partitioned Layout: (n_heads * max_seq_len * head_dim)
// Each head h has a contiguous (max_seq_len x head_dim) buffer enabling
// single-call Accelerate BLAS cblas_sgemv matrix-vector operations.
// ============================================================================
struct KVCache {
    Tensor k;          // Key cache buffer (n_heads * max_seq_len * head_dim)
    Tensor v;          // Value cache buffer (n_heads * max_seq_len * head_dim)
    size_t length{0};  // Current number of cached token states
    size_t n_heads{0};
    size_t head_dim{0};
    size_t max_seq_len{0};

    void clear() noexcept { length = 0; }
    [[nodiscard]] bool empty() const noexcept { return length == 0; }
};

// ============================================================================
// Causal Multi-Head Self-Attention Core
// ============================================================================
class CausalAttention {
public:
    size_t d_model{0};
    size_t n_heads{0};
    size_t max_seq_len{0};
    size_t head_dim{0}; // d_k = d_model / n_heads
    float scale{0.0f};  // 1.0 / sqrt(head_dim)

    // Projection weights and biases
    Tensor W_q; // Query weight (d_model x d_model)
    Tensor b_q; // Query bias   (1 x d_model)
    Tensor W_k; // Key weight   (d_model x d_model)
    Tensor b_k; // Key bias     (1 x d_model)
    Tensor W_v; // Value weight (d_model x d_model)
    Tensor b_v; // Value bias   (1 x d_model)
    Tensor W_o; // Output weight(d_model x d_model)
    Tensor b_o; // Output bias  (1 x d_model)

    // Forward pass cache for autograd backward pass
    AttentionCache cache;

    // --- Lifecycle ---
    CausalAttention() = default;
    CausalAttention(size_t d_model, size_t n_heads, size_t max_seq_len);

    ~CausalAttention() = default;
    CausalAttention(const CausalAttention&) = default;
    CausalAttention& operator=(const CausalAttention&) = default;
    CausalAttention(CausalAttention&&) noexcept = default;
    CausalAttention& operator=(CausalAttention&&) noexcept = default;

    // --- Forward & Backward Core ---

    // Forward Causal Attention:
    // Calculates Q, K, V, per-head scaled causal dot-product scores (Q x K^T / sqrt(d_k)),
    // applies upper-triangular causal mask (-1e9f for j > i), Softmax probabilities,
    // head-value dot-product (A x V), head concatenation, and final output projection (W_o).
    [[nodiscard]] Tensor forward(const Tensor& X);

    // Forward Causal Attention with KV-Cache (Prefill Phase):
    // Computes Q, K, V for all prompt tokens, stores K and V into kv_cache,
    // and returns causal attention output (T x d_model)
    [[nodiscard]] Tensor forward_prefill(const Tensor& X, KVCache& kv_cache);

    // Forward Causal Attention with KV-Cache (Decode Phase):
    // Computes q, k, v for a single new token, appends k and v to kv_cache at row pos,
    // attends over cached keys and values, and returns output (1 x d_model) in O(pos) time
    [[nodiscard]] Tensor forward_decode(const Tensor& X_single, KVCache& kv_cache, size_t pos);

    // Backward Causal Attention:
    // Propagates upstream gradients through W_o, concatenation, per-head values,
    // softmax causal derivatives, scaled dot-product, and input projections (W_q, W_k, W_v).
    // Accumulates parameter gradients and outputs grad_X with respect to input X.
    void backward(const Tensor& grad_output, Tensor& grad_X);

    // --- Memory Management & Initialization ---
    void init_weights(uint64_t seed = 42);
    void zero_gradients() noexcept;
    void clear_cache() noexcept;

    [[nodiscard]] size_t parameter_count() const noexcept {
        return W_q.size() + b_q.size() +
               W_k.size() + b_k.size() +
               W_v.size() + b_v.size() +
               W_o.size() + b_o.size();
    }
};

// ============================================================================
// Feed-Forward Network Cache (Backpropagation Cache)
// ============================================================================
struct FeedForwardCache {
    Tensor X;     // Input activations (T x d_model)
    Tensor H1;    // Intermediate linear activations (T x d_ff)
    Tensor H_act; // Post-ReLU activations (T x d_ff)
};

// ============================================================================
// Multi-Layer Feed-Forward Network (FFN / MLP)
// ============================================================================
class FeedForward {
public:
    size_t d_model{0};
    size_t d_ff{0};

    // Layer 1 Projections
    Tensor W_1; // (d_model x d_ff)
    Tensor b_1; // (1 x d_ff)

    // Layer 2 Projections
    Tensor W_2; // (d_ff x d_model)
    Tensor b_2; // (1 x d_model)

    // Autograd forward cache
    FeedForwardCache cache;

    // --- Lifecycle ---
    FeedForward() = default;
    FeedForward(size_t d_model, size_t d_ff = 0);

    ~FeedForward() = default;
    FeedForward(const FeedForward&) = default;
    FeedForward& operator=(const FeedForward&) = default;
    FeedForward(FeedForward&&) noexcept = default;
    FeedForward& operator=(FeedForward&&) noexcept = default;

    // --- Forward & Backward Core ---

    // Forward pass: Y = W_2 x ReLU(W_1 x X + b_1) + b_2
    [[nodiscard]] Tensor forward(const Tensor& X);

    // Fast decode pass (no autograd caching, in-place SIMD ReLU, zero heap temporaries)
    [[nodiscard]] Tensor forward_decode(const Tensor& X_single);

    // Backward pass: computes analytical gradients through Layer 2, ReLU, and Layer 1
    void backward(const Tensor& grad_output, Tensor& grad_X);

    // --- Memory Management & Initialization ---
    void init_weights(uint64_t seed = 42);
    void zero_gradients() noexcept;
    void clear_cache() noexcept;

    [[nodiscard]] size_t parameter_count() const noexcept {
        return W_1.size() + b_1.size() + W_2.size() + b_2.size();
    }
};

} // namespace titan

#endif // TITAN_ATTENTION_HPP

