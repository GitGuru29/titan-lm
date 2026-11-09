#include "attention.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <cassert>
#include <Accelerate/Accelerate.h>
#include <dispatch/dispatch.h>

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

    // 1. Compute Linear Projections: Q, K, V in parallel via GCD
    cache.Q = Tensor(T, d_model);
    cache.K = Tensor(T, d_model);
    cache.V = Tensor(T, d_model);

    float* q_dest = cache.Q.data_ptr();
    float* k_dest = cache.K.data_ptr();
    float* v_dest = cache.V.data_ptr();
    const float* x_src = X.data_ptr();
    const float* wq_src = W_q.data_ptr();
    const float* wk_src = W_k.data_ptr();
    const float* wv_src = W_v.data_ptr();
    const float* bq_src = b_q.data_ptr();
    const float* bk_src = b_k.data_ptr();
    const float* bv_src = b_v.data_ptr();

    dispatch_queue_t queue = dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);
    dispatch_apply(3, queue, ^(size_t idx) {
        if (idx == 0) {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        static_cast<int>(T), static_cast<int>(d_model), static_cast<int>(d_model),
                        1.0f, x_src, static_cast<int>(d_model),
                        wq_src, static_cast<int>(d_model),
                        0.0f, q_dest, static_cast<int>(d_model));
            for (size_t r = 0; r < T; ++r) {
                vDSP_vadd(q_dest + r * d_model, 1, bq_src, 1, q_dest + r * d_model, 1, d_model);
            }
        } else if (idx == 1) {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        static_cast<int>(T), static_cast<int>(d_model), static_cast<int>(d_model),
                        1.0f, x_src, static_cast<int>(d_model),
                        wk_src, static_cast<int>(d_model),
                        0.0f, k_dest, static_cast<int>(d_model));
            for (size_t r = 0; r < T; ++r) {
                vDSP_vadd(k_dest + r * d_model, 1, bk_src, 1, k_dest + r * d_model, 1, d_model);
            }
        } else {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        static_cast<int>(T), static_cast<int>(d_model), static_cast<int>(d_model),
                        1.0f, x_src, static_cast<int>(d_model),
                        wv_src, static_cast<int>(d_model),
                        0.0f, v_dest, static_cast<int>(d_model));
            for (size_t r = 0; r < T; ++r) {
                vDSP_vadd(v_dest + r * d_model, 1, bv_src, 1, v_dest + r * d_model, 1, d_model);
            }
        }
    });

    // 2. Multi-Head Scaled Dot-Product Attention with Strict Causal Masking
    cache.Q_heads.resize(n_heads);
    cache.K_heads.resize(n_heads);
    cache.V_heads.resize(n_heads);
    cache.scores.resize(n_heads);
    cache.weights.resize(n_heads);
    cache.head_outputs.resize(n_heads);

    Tensor concat_heads(T, d_model, 0.0f);
    float* concat_ptr = concat_heads.data_ptr();
    const float* Q_ptr = cache.Q.data_ptr();
    const float* K_ptr = cache.K.data_ptr();
    const float* V_ptr = cache.V.data_ptr();
    const size_t head_bytes = head_dim * sizeof(float);
    dispatch_apply(n_heads, queue, ^(size_t h) {
        Tensor Q_h(T, head_dim);
        Tensor K_h(T, head_dim);
        Tensor V_h(T, head_dim);

        // Vectorized extraction of head slices
        for (size_t t = 0; t < T; ++t) {
            std::memcpy(Q_h.data_ptr() + t * head_dim, Q_ptr + t * d_model + h * head_dim, head_bytes);
            std::memcpy(K_h.data_ptr() + t * head_dim, K_ptr + t * d_model + h * head_dim, head_bytes);
            std::memcpy(V_h.data_ptr() + t * head_dim, V_ptr + t * d_model + h * head_dim, head_bytes);
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

        // Vectorized assemble into contiguous concatenated head tensor
        for (size_t t = 0; t < T; ++t) {
            std::memcpy(concat_ptr + t * d_model + h * head_dim, head_out_h.data_ptr() + t * head_dim, head_bytes);
        }

        cache.Q_heads[h] = std::move(Q_h);
        cache.K_heads[h] = std::move(K_h);
        cache.V_heads[h] = std::move(V_h);
        cache.scores[h] = std::move(scores_h);
        cache.weights[h] = std::move(weights_h);
        cache.head_outputs[h] = std::move(head_out_h);
    });

    cache.concat_heads = concat_heads;

    // 3. Final Linear Output Projection: Y = concat_heads x W_o + b_o
    Tensor out_proj = Tensor::matmul(concat_heads, W_o);
    return Tensor::add_bias(out_proj, b_o);
}

Tensor CausalAttention::forward_prefill(const Tensor& X, KVCache& kv_cache) {
    if (X.cols != d_model) {
        throw std::invalid_argument("Input tensor feature dimension must match d_model");
    }

    const size_t T = X.rows;
    if (T > max_seq_len) {
        throw std::invalid_argument("Sequence length exceeds max_seq_len in forward_prefill");
    }

    // 1. Compute Linear Projections: Q, K, V in parallel via GCD
    Tensor Q(T, d_model);
    Tensor K(T, d_model);
    Tensor V(T, d_model);

    float* q_dest = Q.data_ptr();
    float* k_dest = K.data_ptr();
    float* v_dest = V.data_ptr();
    const float* x_src = X.data_ptr();
    const float* wq_src = W_q.data_ptr();
    const float* wk_src = W_k.data_ptr();
    const float* wv_src = W_v.data_ptr();
    const float* bq_src = b_q.data_ptr();
    const float* bk_src = b_k.data_ptr();
    const float* bv_src = b_v.data_ptr();

    dispatch_queue_t queue = dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);
    dispatch_apply(3, queue, ^(size_t idx) {
        if (idx == 0) {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        static_cast<int>(T), static_cast<int>(d_model), static_cast<int>(d_model),
                        1.0f, x_src, static_cast<int>(d_model),
                        wq_src, static_cast<int>(d_model),
                        0.0f, q_dest, static_cast<int>(d_model));
            for (size_t r = 0; r < T; ++r) {
                vDSP_vadd(q_dest + r * d_model, 1, bq_src, 1, q_dest + r * d_model, 1, d_model);
            }
        } else if (idx == 1) {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        static_cast<int>(T), static_cast<int>(d_model), static_cast<int>(d_model),
                        1.0f, x_src, static_cast<int>(d_model),
                        wk_src, static_cast<int>(d_model),
                        0.0f, k_dest, static_cast<int>(d_model));
            for (size_t r = 0; r < T; ++r) {
                vDSP_vadd(k_dest + r * d_model, 1, bk_src, 1, k_dest + r * d_model, 1, d_model);
            }
        } else {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        static_cast<int>(T), static_cast<int>(d_model), static_cast<int>(d_model),
                        1.0f, x_src, static_cast<int>(d_model),
                        wv_src, static_cast<int>(d_model),
                        0.0f, v_dest, static_cast<int>(d_model));
            for (size_t r = 0; r < T; ++r) {
                vDSP_vadd(v_dest + r * d_model, 1, bv_src, 1, v_dest + r * d_model, 1, d_model);
            }
        }
    });

    // 2. Cache K and V in contiguous head-major layout: (n_heads * max_seq_len * head_dim)
    const size_t total_cache_elements = n_heads * max_seq_len * head_dim;
    if (kv_cache.k.size() != total_cache_elements || kv_cache.v.size() != total_cache_elements) {
        kv_cache.n_heads = n_heads;
        kv_cache.head_dim = head_dim;
        kv_cache.max_seq_len = max_seq_len;
        kv_cache.k = Tensor(n_heads * max_seq_len, head_dim, 0.0f);
        kv_cache.v = Tensor(n_heads * max_seq_len, head_dim, 0.0f);
    }
    kv_cache.length = T;

    const size_t head_stride = max_seq_len * head_dim;
    const size_t head_bytes = head_dim * sizeof(float);
    float* k_base = kv_cache.k.data_ptr();
    float* v_base = kv_cache.v.data_ptr();
    const float* K_ptr = K.data_ptr();
    const float* V_ptr = V.data_ptr();

    for (size_t h = 0; h < n_heads; ++h) {
        float* k_head = k_base + h * head_stride;
        float* v_head = v_base + h * head_stride;
        for (size_t t = 0; t < T; ++t) {
            std::memcpy(k_head + t * head_dim, K_ptr + t * d_model + h * head_dim, head_bytes);
            std::memcpy(v_head + t * head_dim, V_ptr + t * d_model + h * head_dim, head_bytes);
        }
    }

    // 3. Multi-Head Attention via Parallel GCD
    Tensor concat_heads(T, d_model, 0.0f);
    float* concat_ptr = concat_heads.data_ptr();
    const float* Q_ptr = Q.data_ptr();

    dispatch_apply(n_heads, queue, ^(size_t h) {
        Tensor Q_h(T, head_dim);
        for (size_t t = 0; t < T; ++t) {
            std::memcpy(Q_h.data_ptr() + t * head_dim, Q_ptr + t * d_model + h * head_dim, head_bytes);
        }

        const float* K_h = k_base + h * head_stride;
        const float* V_h = v_base + h * head_stride;

        // AMX Scaled Dot-Product: Scores = (Q_h x K_h^T) * scale
        Tensor scores_h(T, T, 0.0f);
        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasTrans,
            static_cast<int>(T), static_cast<int>(T), static_cast<int>(head_dim),
            scale,
            Q_h.data_ptr(), static_cast<int>(head_dim),
            K_h, static_cast<int>(head_dim),
            0.0f,
            scores_h.data_ptr(), static_cast<int>(T)
        );

        // Strict upper-triangular causal masking
        for (size_t r = 0; r < T; ++r) {
            for (size_t c = r + 1; c < T; ++c) {
                scores_h(r, c) = -1e9f;
            }
        }

        Tensor weights_h = Tensor::forward_softmax(scores_h);

        // AMX Head Context Output: Out_h = Weights_h x V_h
        Tensor head_out_h(T, head_dim, 0.0f);
        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            static_cast<int>(T), static_cast<int>(head_dim), static_cast<int>(T),
            1.0f,
            weights_h.data_ptr(), static_cast<int>(T),
            V_h, static_cast<int>(head_dim),
            0.0f,
            head_out_h.data_ptr(), static_cast<int>(head_dim)
        );

        for (size_t t = 0; t < T; ++t) {
            std::memcpy(concat_ptr + t * d_model + h * head_dim, head_out_h.data_ptr() + t * head_dim, head_bytes);
        }
    });

    Tensor out_proj = Tensor::matmul(concat_heads, W_o);
    return Tensor::add_bias(out_proj, b_o);
}

Tensor CausalAttention::forward_decode(const Tensor& X_single, KVCache& kv_cache, size_t pos) {
    if (X_single.rows != 1 || X_single.cols != d_model) {
        throw std::invalid_argument("X_single must be of shape 1 x d_model");
    }
    if (pos >= max_seq_len) {
        throw std::invalid_argument("pos exceeds max_seq_len in forward_decode");
    }

    // 1. Single Token Projections
    Tensor q_proj = Tensor::matmul(X_single, W_q);
    Tensor q = Tensor::add_bias(q_proj, b_q);

    Tensor k_proj = Tensor::matmul(X_single, W_k);
    Tensor k = Tensor::add_bias(k_proj, b_k);

    Tensor v_proj = Tensor::matmul(X_single, W_v);
    Tensor v = Tensor::add_bias(v_proj, b_v);

    // 2. Append new k, v token slices into head-major contiguous cache
    const size_t head_stride = max_seq_len * head_dim;
    const size_t head_bytes = head_dim * sizeof(float);
    float* k_base = kv_cache.k.data_ptr();
    float* v_base = kv_cache.v.data_ptr();
    const float* k_src = k.data_ptr();
    const float* v_src = v.data_ptr();

    for (size_t h = 0; h < n_heads; ++h) {
        std::memcpy(k_base + h * head_stride + pos * head_dim, k_src + h * head_dim, head_bytes);
        std::memcpy(v_base + h * head_stride + pos * head_dim, v_src + h * head_dim, head_bytes);
    }
    kv_cache.length = pos + 1;

    const size_t total_len = pos + 1;
    Tensor concat_heads(1, d_model, 0.0f);
    float* concat_ptr = concat_heads.data_ptr();
    const float* q_ptr = q.data_ptr();

    // 3. Multi-head decode attention using parallel GCD & Accelerate BLAS GEMV
    dispatch_queue_t queue = dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);
    dispatch_apply(n_heads, queue, ^(size_t h) {
        std::vector<float> scores(total_len);
        std::vector<float> weights(total_len);
        const float* q_h = q_ptr + h * head_dim;
        const float* K_h = k_base + h * head_stride;
        const float* V_h = v_base + h * head_stride;

        // Scores = (K_h x Q_h) * scale -> (total_len x 1)
        cblas_sgemv(
            CblasRowMajor, CblasNoTrans,
            static_cast<int>(total_len), static_cast<int>(head_dim),
            scale,
            K_h, static_cast<int>(head_dim),
            q_h, 1,
            0.0f,
            scores.data(), 1
        );

        // Vectorized stable softmax
        float max_score = -1e9f;
        vDSP_maxv(scores.data(), 1, &max_score, total_len);
        float neg_max = -max_score;
        vDSP_vsadd(scores.data(), 1, &neg_max, weights.data(), 1, total_len);
        int len_i = static_cast<int>(total_len);
        vvexpf(weights.data(), weights.data(), &len_i);
        float sum_exp = 0.0f;
        vDSP_sve(weights.data(), 1, &sum_exp, total_len);
        float inv_sum = 1.0f / (sum_exp > 0.0f ? sum_exp : 1e-12f);
        vDSP_vsmul(weights.data(), 1, &inv_sum, weights.data(), 1, total_len);

        // Output context = Weights x V_h -> directly into concat_heads slice!
        cblas_sgemv(
            CblasRowMajor, CblasTrans,
            static_cast<int>(total_len), static_cast<int>(head_dim),
            1.0f,
            V_h, static_cast<int>(head_dim),
            weights.data(), 1,
            0.0f,
            concat_ptr + h * head_dim, 1
        );
    });

    // 4. Output Projection: Y = concat_heads x W_o + b_o
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

    const float* dY_ptr = grad_output.resolve_read_ptr();

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
    float* dQ_ptr = dQ.data_ptr();
    float* dK_ptr = dK.data_ptr();
    float* dV_ptr = dV.data_ptr();
    const float* d_concat_ptr = d_concat.data_ptr();
    const size_t head_bytes = head_dim * sizeof(float);
    dispatch_queue_t queue = dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);
    dispatch_apply(n_heads, queue, ^(size_t h) {
        Tensor d_head_out_h(T, head_dim);
        for (size_t t = 0; t < T; ++t) {
            std::memcpy(d_head_out_h.data_ptr() + t * head_dim, d_concat_ptr + t * d_model + h * head_dim, head_bytes);
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
            std::memcpy(dQ_ptr + t * d_model + h * head_dim, dQ_h.data_ptr() + t * head_dim, head_bytes);
            std::memcpy(dK_ptr + t * d_model + h * head_dim, dK_h.data_ptr() + t * head_dim, head_bytes);
            std::memcpy(dV_ptr + t * d_model + h * head_dim, dV_h.data_ptr() + t * head_dim, head_bytes);
        }
    });

    // 3. Input Projections Backward (Q, K, V -> X)
    // Run dW and db parameter updates for Q, K, V in parallel across cores via GCD
    const float* x_ptr = cache.X.data_ptr();
    const float* dq_ptr = dQ.data_ptr();
    const float* dk_ptr = dK.data_ptr();
    const float* dv_ptr = dV.data_ptr();
    float* bq_grad = b_q.grad_ptr();
    float* bk_grad = b_k.grad_ptr();
    float* bv_grad = b_v.grad_ptr();
    float* wq_grad = W_q.grad_ptr();
    float* wk_grad = W_k.grad_ptr();
    float* wv_grad = W_v.grad_ptr();

    dispatch_apply(3, queue, ^(size_t idx) {
        if (idx == 0) {
            for (size_t r = 0; r < T; ++r) {
                vDSP_vadd(bq_grad, 1, dq_ptr + r * d_model, 1, bq_grad, 1, d_model);
            }
            cblas_sgemm(
                CblasRowMajor, CblasTrans, CblasNoTrans,
                static_cast<int>(d_model), static_cast<int>(d_model), static_cast<int>(T),
                1.0f, x_ptr, static_cast<int>(d_model),
                dq_ptr, static_cast<int>(d_model),
                1.0f, wq_grad, static_cast<int>(d_model)
            );
        } else if (idx == 1) {
            for (size_t r = 0; r < T; ++r) {
                vDSP_vadd(bk_grad, 1, dk_ptr + r * d_model, 1, bk_grad, 1, d_model);
            }
            cblas_sgemm(
                CblasRowMajor, CblasTrans, CblasNoTrans,
                static_cast<int>(d_model), static_cast<int>(d_model), static_cast<int>(T),
                1.0f, x_ptr, static_cast<int>(d_model),
                dk_ptr, static_cast<int>(d_model),
                1.0f, wk_grad, static_cast<int>(d_model)
            );
        } else {
            for (size_t r = 0; r < T; ++r) {
                vDSP_vadd(bv_grad, 1, dv_ptr + r * d_model, 1, bv_grad, 1, d_model);
            }
            cblas_sgemm(
                CblasRowMajor, CblasTrans, CblasNoTrans,
                static_cast<int>(d_model), static_cast<int>(d_model), static_cast<int>(T),
                1.0f, x_ptr, static_cast<int>(d_model),
                dv_ptr, static_cast<int>(d_model),
                1.0f, wv_grad, static_cast<int>(d_model)
            );
        }
    });

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

    const float* dY_ptr = grad_output.resolve_read_ptr();

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

