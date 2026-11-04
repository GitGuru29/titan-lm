#include "attention.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <cassert>
#include <Accelerate/Accelerate.h>

namespace titan {

// ============================================================================
// CausalAttention Implementation
// ============================================================================

CausalAttention::CausalAttention(size_t d_model, size_t n_heads, size_t max_seq_len)
    : d_model(d_model), n_heads(n_heads), max_seq_len(max_seq_len) {
    if (n_heads == 0 || d_model % n_heads != 0) {
        throw std::invalid_argument(
            "d_model (" + std::to_string(d_model) +
            ") must be evenly divisible by n_heads (" + std::to_string(n_heads) + ")"
        );
    }

    head_dim = d_model / n_heads;
    scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Allocate projection weights and biases
    W_q = Tensor(d_model, d_model);
    b_q = Tensor(1, d_model);

    W_k = Tensor(d_model, d_model);
    b_k = Tensor(1, d_model);

    W_v = Tensor(d_model, d_model);
    b_v = Tensor(1, d_model);

    W_o = Tensor(d_model, d_model);
    b_o = Tensor(1, d_model);

    init_weights(42);
}

void CausalAttention::init_weights(uint64_t seed) {
    W_q.init_xavier(d_model, d_model, seed + 1);
    b_q.zero_data();

    W_k.init_xavier(d_model, d_model, seed + 2);
    b_k.zero_data();

    W_v.init_xavier(d_model, d_model, seed + 3);
    b_v.zero_data();

    W_o.init_xavier(d_model, d_model, seed + 4);
    b_o.zero_data();

    zero_gradients();
}

void CausalAttention::zero_gradients() noexcept {
    W_q.zero_gradients();
    b_q.zero_gradients();

    W_k.zero_gradients();
    b_k.zero_gradients();

    W_v.zero_gradients();
    b_v.zero_gradients();

    W_o.zero_gradients();
    b_o.zero_gradients();
}

void CausalAttention::clear_cache() noexcept {
    cache.X = Tensor();
    cache.Q = Tensor();
    cache.K = Tensor();
    cache.V = Tensor();
    cache.Q_heads.clear();
    cache.K_heads.clear();
    cache.V_heads.clear();
    cache.scores.clear();
    cache.weights.clear();
    cache.head_outputs.clear();
    cache.concat_heads = Tensor();
}

Tensor CausalAttention::forward(const Tensor& X) {
    if (X.cols != d_model) {
        throw std::invalid_argument(
            "Input tensor feature dimension (" + std::to_string(X.cols) +
            ") must match d_model (" + std::to_string(d_model) + ")"
        );
    }

    const size_t T = X.rows;
    if (T > max_seq_len) {
        throw std::invalid_argument(
            "Input sequence length (" + std::to_string(T) +
            ") exceeds configured max_seq_len (" + std::to_string(max_seq_len) + ")"
        );
    }

    cache.X = X;

    // 1. Compute Linear Projections: Q, K, V
    Tensor Q_proj = Tensor::matmul(X, W_q);
    cache.Q = Tensor::add_bias(Q_proj, b_q);

    Tensor K_proj = Tensor::matmul(X, W_k);
    cache.K = Tensor::add_bias(K_proj, b_k);

    Tensor V_proj = Tensor::matmul(X, W_v);
    cache.V = Tensor::add_bias(V_proj, b_v);

    // 2. Multi-Head Scaled Dot-Product Attention with Strict Causal Masking
    cache.Q_heads.resize(n_heads);
    cache.K_heads.resize(n_heads);
    cache.V_heads.resize(n_heads);
    cache.scores.resize(n_heads);
    cache.weights.resize(n_heads);
    cache.head_outputs.resize(n_heads);

    Tensor concat_heads(T, d_model, 0.0f);

    for (size_t h = 0; h < n_heads; ++h) {
        Tensor Q_h(T, head_dim);
        Tensor K_h(T, head_dim);
        Tensor V_h(T, head_dim);

        // Extract head slices
        for (size_t t = 0; t < T; ++t) {
            for (size_t c = 0; c < head_dim; ++c) {
                Q_h(t, c) = cache.Q(t, h * head_dim + c);
                K_h(t, c) = cache.K(t, h * head_dim + c);
                V_h(t, c) = cache.V(t, h * head_dim + c);
            }
        }

        // AMX Scaled Dot-Product: Scores = (Q_h x K_h^T) * scale
        Tensor scores_h(T, T, 0.0f);
        cblas_sgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasTrans,
            static_cast<int>(T),
            static_cast<int>(T),
            static_cast<int>(head_dim),
            scale,
            Q_h.data_ptr(),
            static_cast<int>(head_dim),
            K_h.data_ptr(),
            static_cast<int>(head_dim),
            0.0f,
            scores_h.data_ptr(),
            static_cast<int>(T)
        );

        // Strict upper-triangular causal masking: tokens cannot attend to future tokens (j > i)
        for (size_t r = 0; r < T; ++r) {
            for (size_t c = r + 1; c < T; ++c) {
                scores_h(r, c) = -1e9f;
            }
        }

        // Row-wise Softmax over causal scores
        Tensor weights_h = Tensor::forward_softmax(scores_h);

        // AMX Head Context Output: Out_h = Weights_h x V_h
        Tensor head_out_h(T, head_dim, 0.0f);
        cblas_sgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasNoTrans,
            static_cast<int>(T),
            static_cast<int>(head_dim),
            static_cast<int>(T),
            1.0f,
            weights_h.data_ptr(),
            static_cast<int>(T),
            V_h.data_ptr(),
            static_cast<int>(head_dim),
            0.0f,
            head_out_h.data_ptr(),
            static_cast<int>(head_dim)
        );

        // Assemble into contiguous concatenated head tensor
        for (size_t t = 0; t < T; ++t) {
            for (size_t c = 0; c < head_dim; ++c) {
                concat_heads(t, h * head_dim + c) = head_out_h(t, c);
            }
        }

        cache.Q_heads[h] = std::move(Q_h);
        cache.K_heads[h] = std::move(K_h);
        cache.V_heads[h] = std::move(V_h);
        cache.scores[h] = std::move(scores_h);
        cache.weights[h] = std::move(weights_h);
        cache.head_outputs[h] = std::move(head_out_h);
    }

    cache.concat_heads = concat_heads;

    // 3. Final Linear Output Projection: Y = concat_heads x W_o + b_o
    Tensor out_proj = Tensor::matmul(concat_heads, W_o);
    return Tensor::add_bias(out_proj, b_o);
}

void CausalAttention::backward(const Tensor& grad_output, Tensor& grad_X) {
    const size_t T = cache.X.rows;
    if (grad_output.rows != T || grad_output.cols != d_model) {
        throw std::invalid_argument("grad_output shape mismatch in CausalAttention::backward");
    }
    if (grad_X.rows != T || grad_X.cols != d_model) {
        throw std::invalid_argument("grad_X shape mismatch in CausalAttention::backward");
    }

    const float* dY_ptr = grad_output.data_ptr();

    // 1. Output Projection Backward
    Tensor d_concat(T, d_model, 0.0f);

    // db_o += col_sums(dY)
    for (size_t r = 0; r < T; ++r) {
        vDSP_vadd(b_o.grad_ptr(), 1, dY_ptr + r * d_model, 1, b_o.grad_ptr(), 1, d_model);
    }

    // dW_o += concat_heads^T x dY
    cblas_sgemm(
        CblasRowMajor,
        CblasTrans,
        CblasNoTrans,
        static_cast<int>(d_model),
        static_cast<int>(d_model),
        static_cast<int>(T),
        1.0f,
        cache.concat_heads.data_ptr(),
        static_cast<int>(d_model),
        dY_ptr,
        static_cast<int>(d_model),
        1.0f, // Accumulate
        W_o.grad_ptr(),
        static_cast<int>(d_model)
    );

    // d_concat = dY x W_o^T
    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasTrans,
        static_cast<int>(T),
        static_cast<int>(d_model),
        static_cast<int>(d_model),
        1.0f,
        dY_ptr,
        static_cast<int>(d_model),
        W_o.data_ptr(),
        static_cast<int>(d_model),
        0.0f,
        d_concat.data_ptr(),
        static_cast<int>(d_model)
    );

    // 2. Multi-Head Scaled Dot-Product Backward Loops
    Tensor dQ(T, d_model, 0.0f);
    Tensor dK(T, d_model, 0.0f);
    Tensor dV(T, d_model, 0.0f);

    for (size_t h = 0; h < n_heads; ++h) {
        Tensor d_head_out_h(T, head_dim);
        for (size_t t = 0; t < T; ++t) {
            for (size_t c = 0; c < head_dim; ++c) {
                d_head_out_h(t, c) = d_concat(t, h * head_dim + c);
            }
        }

        // a. dV_h = weights_h^T x d_head_out_h
        Tensor dV_h(T, head_dim, 0.0f);
        cblas_sgemm(
            CblasRowMajor,
            CblasTrans,
            CblasNoTrans,
            static_cast<int>(T),
            static_cast<int>(head_dim),
            static_cast<int>(T),
            1.0f,
            cache.weights[h].data_ptr(),
            static_cast<int>(T),
            d_head_out_h.data_ptr(),
            static_cast<int>(head_dim),
            0.0f,
            dV_h.data_ptr(),
            static_cast<int>(head_dim)
        );

        // b. d_weights_h = d_head_out_h x V_h^T
        Tensor d_weights_h(T, T, 0.0f);
        cblas_sgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasTrans,
            static_cast<int>(T),
            static_cast<int>(T),
            static_cast<int>(head_dim),
            1.0f,
            d_head_out_h.data_ptr(),
            static_cast<int>(head_dim),
            cache.V_heads[h].data_ptr(),
            static_cast<int>(head_dim),
            0.0f,
            d_weights_h.data_ptr(),
            static_cast<int>(T)
        );

        // c. Softmax backward: d_scores_h = d_weights_h x J_softmax
        Tensor d_scores_h(T, T, 0.0f);
        Tensor::backward_softmax(cache.weights[h], d_weights_h, d_scores_h);

        // d. dQ_h = scale * (d_scores_h x K_h)
        Tensor dQ_h(T, head_dim, 0.0f);
        cblas_sgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasNoTrans,
            static_cast<int>(T),
            static_cast<int>(head_dim),
            static_cast<int>(T),
            scale,
            d_scores_h.grad_ptr(),
            static_cast<int>(T),
            cache.K_heads[h].data_ptr(),
            static_cast<int>(head_dim),
            0.0f,
            dQ_h.data_ptr(),
            static_cast<int>(head_dim)
        );

        // e. dK_h = scale * (d_scores_h^T x Q_h)
        Tensor dK_h(T, head_dim, 0.0f);
        cblas_sgemm(
            CblasRowMajor,
            CblasTrans,
            CblasNoTrans,
            static_cast<int>(T),
            static_cast<int>(head_dim),
            static_cast<int>(T),
            scale,
            d_scores_h.grad_ptr(),
            static_cast<int>(T),
            cache.Q_heads[h].data_ptr(),
            static_cast<int>(head_dim),
            0.0f,
            dK_h.data_ptr(),
            static_cast<int>(head_dim)
        );

        // Concatenate back into full dQ, dK, dV tensors
        for (size_t t = 0; t < T; ++t) {
            for (size_t c = 0; c < head_dim; ++c) {
                dQ(t, h * head_dim + c) = dQ_h(t, c);
                dK(t, h * head_dim + c) = dK_h(t, c);
                dV(t, h * head_dim + c) = dV_h(t, c);
            }
        }
    }

    // 3. Input Projections Backward (Q, K, V -> X)

    // db_q += col_sums(dQ)
    for (size_t r = 0; r < T; ++r) {
        vDSP_vadd(b_q.grad_ptr(), 1, dQ.data_ptr() + r * d_model, 1, b_q.grad_ptr(), 1, d_model);
    }
    // dW_q += X^T x dQ
    cblas_sgemm(
        CblasRowMajor, CblasTrans, CblasNoTrans,
        static_cast<int>(d_model), static_cast<int>(d_model), static_cast<int>(T),
        1.0f, cache.X.data_ptr(), static_cast<int>(d_model),
        dQ.data_ptr(), static_cast<int>(d_model),
        1.0f, W_q.grad_ptr(), static_cast<int>(d_model)
    );

    // db_k += col_sums(dK)
    for (size_t r = 0; r < T; ++r) {
        vDSP_vadd(b_k.grad_ptr(), 1, dK.data_ptr() + r * d_model, 1, b_k.grad_ptr(), 1, d_model);
    }
    // dW_k += X^T x dK
    cblas_sgemm(
        CblasRowMajor, CblasTrans, CblasNoTrans,
        static_cast<int>(d_model), static_cast<int>(d_model), static_cast<int>(T),
        1.0f, cache.X.data_ptr(), static_cast<int>(d_model),
        dK.data_ptr(), static_cast<int>(d_model),
        1.0f, W_k.grad_ptr(), static_cast<int>(d_model)
    );

    // db_v += col_sums(dV)
    for (size_t r = 0; r < T; ++r) {
        vDSP_vadd(b_v.grad_ptr(), 1, dV.data_ptr() + r * d_model, 1, b_v.grad_ptr(), 1, d_model);
    }
    // dW_v += X^T x dV
    cblas_sgemm(
        CblasRowMajor, CblasTrans, CblasNoTrans,
        static_cast<int>(d_model), static_cast<int>(d_model), static_cast<int>(T),
        1.0f, cache.X.data_ptr(), static_cast<int>(d_model),
        dV.data_ptr(), static_cast<int>(d_model),
        1.0f, W_v.grad_ptr(), static_cast<int>(d_model)
    );

    // Accumulate dX = dQ x W_q^T + dK x W_k^T + dV x W_v^T into grad_X.grad
    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasTrans,
        static_cast<int>(T), static_cast<int>(d_model), static_cast<int>(d_model),
        1.0f, dQ.data_ptr(), static_cast<int>(d_model),
        W_q.data_ptr(), static_cast<int>(d_model),
        1.0f, grad_X.grad_ptr(), static_cast<int>(d_model)
    );
    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasTrans,
        static_cast<int>(T), static_cast<int>(d_model), static_cast<int>(d_model),
        1.0f, dK.data_ptr(), static_cast<int>(d_model),
        W_k.data_ptr(), static_cast<int>(d_model),
        1.0f, grad_X.grad_ptr(), static_cast<int>(d_model)
    );
    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasTrans,
        static_cast<int>(T), static_cast<int>(d_model), static_cast<int>(d_model),
        1.0f, dV.data_ptr(), static_cast<int>(d_model),
        W_v.data_ptr(), static_cast<int>(d_model),
        1.0f, grad_X.grad_ptr(), static_cast<int>(d_model)
    );
}

// ============================================================================
// FeedForward Implementation
// ============================================================================

FeedForward::FeedForward(size_t d_model, size_t d_ff)
    : d_model(d_model), d_ff(d_ff == 0 ? (4 * d_model) : d_ff) {
    W_1 = Tensor(d_model, this->d_ff);
    b_1 = Tensor(1, this->d_ff);

    W_2 = Tensor(this->d_ff, d_model);
    b_2 = Tensor(1, d_model);

    init_weights(100);
}

void FeedForward::init_weights(uint64_t seed) {
    W_1.init_xavier(d_model, d_ff, seed + 10);
    b_1.zero_data();

    W_2.init_xavier(d_ff, d_model, seed + 20);
    b_2.zero_data();

    zero_gradients();
}

void FeedForward::zero_gradients() noexcept {
    W_1.zero_gradients();
    b_1.zero_gradients();
    W_2.zero_gradients();
    b_2.zero_gradients();
}

void FeedForward::clear_cache() noexcept {
    cache.X = Tensor();
    cache.H1 = Tensor();
    cache.H_act = Tensor();
}

Tensor FeedForward::forward(const Tensor& X) {
    if (X.cols != d_model) {
        throw std::invalid_argument(
            "Input feature dimension (" + std::to_string(X.cols) +
            ") must match FeedForward d_model (" + std::to_string(d_model) + ")"
        );
    }

    cache.X = X;

    // 1. Layer 1: H1 = X x W_1 + b_1
    Tensor H1_proj = Tensor::matmul(X, W_1);
    cache.H1 = Tensor::add_bias(H1_proj, b_1);

    // 2. Intermediate ReLU Activation: H_act = max(0, H1)
    cache.H_act = Tensor::forward_relu(cache.H1);

    // 3. Layer 2: Y = H_act x W_2 + b_2
    Tensor H2_proj = Tensor::matmul(cache.H_act, W_2);
    return Tensor::add_bias(H2_proj, b_2);
}

void FeedForward::backward(const Tensor& grad_output, Tensor& grad_X) {
    const size_t T = cache.X.rows;
    if (grad_output.rows != T || grad_output.cols != d_model) {
        throw std::invalid_argument("grad_output shape mismatch in FeedForward::backward");
    }
    if (grad_X.rows != T || grad_X.cols != d_model) {
        throw std::invalid_argument("grad_X shape mismatch in FeedForward::backward");
    }

    const float* dY_ptr = grad_output.data_ptr();

    // 1. Layer 2 Backward
    // db_2 += col_sums(dY)
    for (size_t r = 0; r < T; ++r) {
        vDSP_vadd(b_2.grad_ptr(), 1, dY_ptr + r * d_model, 1, b_2.grad_ptr(), 1, d_model);
    }

    // dW_2 += H_act^T x dY
    cblas_sgemm(
        CblasRowMajor, CblasTrans, CblasNoTrans,
        static_cast<int>(d_ff), static_cast<int>(d_model), static_cast<int>(T),
        1.0f, cache.H_act.data_ptr(), static_cast<int>(d_ff),
        dY_ptr, static_cast<int>(d_model),
        1.0f, W_2.grad_ptr(), static_cast<int>(d_model)
    );

    // dH_act = dY x W_2^T
    Tensor dH_act(T, d_ff, 0.0f);
    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasTrans,
        static_cast<int>(T), static_cast<int>(d_ff), static_cast<int>(d_model),
        1.0f, dY_ptr, static_cast<int>(d_model),
        W_2.data_ptr(), static_cast<int>(d_model),
        0.0f, dH_act.data_ptr(), static_cast<int>(d_ff)
    );

    // 2. Intermediate ReLU Backward
    Tensor dH1(T, d_ff, 0.0f);
    Tensor::backward_relu(cache.H1, dH_act, dH1);

    // 3. Layer 1 Backward
    // db_1 += col_sums(dH1)
    for (size_t r = 0; r < T; ++r) {
        vDSP_vadd(b_1.grad_ptr(), 1, dH1.grad_ptr() + r * d_ff, 1, b_1.grad_ptr(), 1, d_ff);
    }

    // dW_1 += X^T x dH1
    cblas_sgemm(
        CblasRowMajor, CblasTrans, CblasNoTrans,
        static_cast<int>(d_model), static_cast<int>(d_ff), static_cast<int>(T),
        1.0f, cache.X.data_ptr(), static_cast<int>(d_model),
        dH1.grad_ptr(), static_cast<int>(d_ff),
        1.0f, W_1.grad_ptr(), static_cast<int>(d_ff)
    );

    // dX += dH1 x W_1^T accumulated directly into grad_X.grad
    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasTrans,
        static_cast<int>(T), static_cast<int>(d_model), static_cast<int>(d_ff),
        1.0f, dH1.grad_ptr(), static_cast<int>(d_ff),
        W_1.data_ptr(), static_cast<int>(d_ff),
        1.0f, grad_X.grad_ptr(), static_cast<int>(d_model)
    );
}

} // namespace titan

