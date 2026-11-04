#include "model.hpp"
#include "tokenizer.hpp"

#include <cmath>
#include <iostream>
#include <fstream>
#include <random>
#include <stdexcept>
#include <cassert>
#include <Accelerate/Accelerate.h>

namespace titan {

// ============================================================================
// LayerNorm Implementation (Apple Silicon vDSP Vectorized)
// ============================================================================

LayerNorm::LayerNorm(size_t d_model, float eps)
    : d_model(d_model), eps(eps) {
    gamma = Tensor(1, d_model, 1.0f);
    beta = Tensor(1, d_model, 0.0f);
    init_weights();
}

void LayerNorm::init_weights() {
    gamma.fill_constant(1.0f);
    beta.fill_constant(0.0f);
    zero_gradients();
}

void LayerNorm::zero_gradients() noexcept {
    gamma.zero_gradients();
    beta.zero_gradients();
}

void LayerNorm::clear_cache() noexcept {
    cache.X = Tensor();
    cache.X_hat = Tensor();
    cache.mean.clear();
    cache.rstd.clear();
}

Tensor LayerNorm::forward(const Tensor& X) {
    if (X.cols != d_model) {
        throw std::invalid_argument(
            "Input feature dimension (" + std::to_string(X.cols) +
            ") must match LayerNorm d_model (" + std::to_string(d_model) + ")"
        );
    }

    const size_t T = X.rows;
    cache.X = X;

    Tensor Y(T, d_model, 0.0f);
    Tensor X_hat(T, d_model, 0.0f);
    cache.mean.resize(T);
    cache.rstd.resize(T);

    const float* gamma_ptr = gamma.data_ptr();
    const float* beta_ptr = beta.data_ptr();

    // Compute normalization per row across d_model leveraging Apple Accelerate vDSP
    for (size_t t = 0; t < T; ++t) {
        const float* x_t = X.data_ptr() + t * d_model;
        float* x_hat_t = X_hat.data_ptr() + t * d_model;
        float* y_t = Y.data_ptr() + t * d_model;

        // 1. Mean: sum(x_t) / d_model
        float sum = 0.0f;
        vDSP_sve(x_t, 1, &sum, d_model);
        float mean = sum / static_cast<float>(d_model);
        cache.mean[t] = mean;

        // 2. Subtract Mean: (x_t - mean) -> x_hat_t
        float neg_mean = -mean;
        vDSP_vsadd(x_t, 1, &neg_mean, x_hat_t, 1, d_model);

        // 3. Variance: dot(x_hat_t, x_hat_t) / d_model
        float sq_sum = 0.0f;
        vDSP_dotpr(x_hat_t, 1, x_hat_t, 1, &sq_sum, d_model);
        float var = sq_sum / static_cast<float>(d_model);
        float rstd = 1.0f / std::sqrt(var + eps);
        cache.rstd[t] = rstd;

        // 4. Normalize: x_hat_t = (x_t - mean) * rstd
        vDSP_vsmul(x_hat_t, 1, &rstd, x_hat_t, 1, d_model);

        // 5. Scale and Shift: y_t = x_hat_t * gamma + beta
        vDSP_vmul(x_hat_t, 1, gamma_ptr, 1, y_t, 1, d_model);
        vDSP_vadd(y_t, 1, beta_ptr, 1, y_t, 1, d_model);
    }

    cache.X_hat = std::move(X_hat);
    return Y;
}

void LayerNorm::backward(const Tensor& grad_output, Tensor& grad_X) {
    const size_t T = cache.X.rows;
    if (grad_output.rows != T || grad_output.cols != d_model) {
        throw std::invalid_argument("grad_output shape mismatch in LayerNorm::backward");
    }
    if (grad_X.rows != T || grad_X.cols != d_model) {
        throw std::invalid_argument("grad_X shape mismatch in LayerNorm::backward");
    }

    const float* gamma_ptr = gamma.data_ptr();
    float* gamma_grad_ptr = gamma.grad_ptr();
    float* beta_grad_ptr = beta.grad_ptr();
    const float* dy_base = grad_output.resolve_read_ptr();

    // Reusable vector buffers for vDSP vector calculations
    std::vector<float> dy_xhat(d_model);
    std::vector<float> dy_gamma(d_model);
    std::vector<float> temp(d_model);

    const float inv_D = 1.0f / static_cast<float>(d_model);

    for (size_t t = 0; t < T; ++t) {
        const float* dy_t = dy_base + t * d_model;
        const float* x_hat_t = cache.X_hat.data_ptr() + t * d_model;
        float* dx_t = grad_X.grad_ptr() + t * d_model;

        // 1. Accumulate dbeta: beta.grad += dy_t
        vDSP_vadd(beta_grad_ptr, 1, dy_t, 1, beta_grad_ptr, 1, d_model);

        // 2. Accumulate dgamma: gamma.grad += dy_t * x_hat_t
        vDSP_vmul(dy_t, 1, x_hat_t, 1, dy_xhat.data(), 1, d_model);
        vDSP_vadd(gamma_grad_ptr, 1, dy_xhat.data(), 1, gamma_grad_ptr, 1, d_model);

        // 3. Analytical input gradient via vDSP vector primitives:
        // dy_gamma = dy_t * gamma
        vDSP_vmul(dy_t, 1, gamma_ptr, 1, dy_gamma.data(), 1, d_model);

        // S1 = sum(dy_gamma)
        float S1 = 0.0f;
        vDSP_sve(dy_gamma.data(), 1, &S1, d_model);

        // S2 = dot(dy_gamma, x_hat_t)
        float S2 = 0.0f;
        vDSP_dotpr(dy_gamma.data(), 1, x_hat_t, 1, &S2, d_model);

        float c1 = S1 * inv_D;
        float c2 = S2 * inv_D;

        // dx = rstd * (dy_gamma - c1 - c2 * x_hat)
        float neg_c2 = -c2;
        vDSP_vsmul(x_hat_t, 1, &neg_c2, temp.data(), 1, d_model);
        vDSP_vadd(temp.data(), 1, dy_gamma.data(), 1, temp.data(), 1, d_model);

        float neg_c1 = -c1;
        vDSP_vsadd(temp.data(), 1, &neg_c1, temp.data(), 1, d_model);

        float rstd_val = cache.rstd[t];
        vDSP_vsmul(temp.data(), 1, &rstd_val, temp.data(), 1, d_model);

        // Accumulate into grad_X.grad
        vDSP_vadd(dx_t, 1, temp.data(), 1, dx_t, 1, d_model);
    }
}

// ============================================================================
// TransformerBlock Implementation (Pre-LN with Dual Residual Connections)
// ============================================================================

TransformerBlock::TransformerBlock(
    size_t d_model,
    size_t n_heads,
    size_t max_seq_len,
    size_t d_ff,
    float eps
) : d_model(d_model),
    n_heads(n_heads),
    d_ff(d_ff == 0 ? (4 * d_model) : d_ff),
    max_seq_len(max_seq_len) {

    ln_1 = LayerNorm(d_model, eps);
    attn = CausalAttention(d_model, n_heads, max_seq_len);
    ln_2 = LayerNorm(d_model, eps);
    ffn  = FeedForward(d_model, this->d_ff);

    init_weights(42);
}

void TransformerBlock::init_weights(uint64_t seed) {
    ln_1.init_weights();
    attn.init_weights(seed + 100);
    ln_2.init_weights();
    ffn.init_weights(seed + 200);
    zero_gradients();
}

void TransformerBlock::zero_gradients() noexcept {
    ln_1.zero_gradients();
    attn.zero_gradients();
    ln_2.zero_gradients();
    ffn.zero_gradients();
}

void TransformerBlock::clear_cache() noexcept {
    cache.X = Tensor();
    cache.X_norm1 = Tensor();
    cache.X_attn = Tensor();
    cache.X_mid = Tensor();
    cache.X_norm2 = Tensor();
    cache.X_ffn = Tensor();

    ln_1.clear_cache();
    attn.clear_cache();
    ln_2.clear_cache();
    ffn.clear_cache();
}

Tensor TransformerBlock::forward(const Tensor& X) {
    if (X.cols != d_model) {
        throw std::invalid_argument(
            "Input feature dimension (" + std::to_string(X.cols) +
            ") must match TransformerBlock d_model (" + std::to_string(d_model) + ")"
        );
    }

    cache.X = X;

    // --- Sub-Layer 1: Causal Self-Attention with Pre-LN & First Residual ---
    // 1. X_norm1 = ln_1(X)
    cache.X_norm1 = ln_1.forward(X);

    // 2. X_attn = attn(X_norm1)
    cache.X_attn = attn.forward(cache.X_norm1);

    // 3. X_mid = X + X_attn (First Residual Stream)
    cache.X_mid = Tensor::add(X, cache.X_attn);

    // --- Sub-Layer 2: Feed-Forward Network with Pre-LN & Second Residual ---
    // 4. X_norm2 = ln_2(X_mid)
    cache.X_norm2 = ln_2.forward(cache.X_mid);

    // 5. X_ffn = ffn(X_norm2)
    cache.X_ffn = ffn.forward(cache.X_norm2);

    // 6. Y = X_mid + X_ffn (Second Residual Stream)
    return Tensor::add(cache.X_mid, cache.X_ffn);
}

void TransformerBlock::backward(const Tensor& grad_output, Tensor& grad_X) {
    const size_t T = cache.X.rows;
    if (grad_output.rows != T || grad_output.cols != d_model) {
        throw std::invalid_argument("grad_output shape mismatch in TransformerBlock::backward");
    }
    if (grad_X.rows != T || grad_X.cols != d_model) {
        throw std::invalid_argument("grad_X shape mismatch in TransformerBlock::backward");
    }

    // --- 1. Second Residual Stream Backward (Y = X_mid + X_ffn) ---
    const float* dY_ptr = grad_output.resolve_read_ptr();

    // Backpropagate through FFN: dX_norm2
    Tensor dX_norm2(T, d_model, 0.0f);
    ffn.backward(grad_output, dX_norm2);

    // Backpropagate through ln_2: dX_mid_ffn
    Tensor dX_mid_ffn(T, d_model, 0.0f);
    ln_2.backward(dX_norm2, dX_mid_ffn);

    // Accumulate gradients at the intermediate residual stream:
    // dX_mid = dY (residual stream) + dX_mid_ffn (FFN branch)
    Tensor dX_mid(T, d_model, 0.0f);
    vDSP_vadd(dY_ptr, 1, dX_mid_ffn.resolve_read_ptr(), 1, dX_mid.data_ptr(), 1, dX_mid.size());

    // --- 2. First Residual Stream Backward (X_mid = X + X_attn) ---
    // Backpropagate through Attention: dX_norm1
    Tensor dX_norm1(T, d_model, 0.0f);
    attn.backward(dX_mid, dX_norm1);

    // Backpropagate through ln_1: dX_ln1
    Tensor dX_ln1(T, d_model, 0.0f);
    ln_1.backward(dX_norm1, dX_ln1);

    // Accumulate total input gradient into grad_X.grad:
    // dX = dX_mid (residual stream) + dX_ln1 (attention branch)
    vDSP_vadd(grad_X.grad_ptr(), 1, dX_mid.data_ptr(), 1, grad_X.grad_ptr(), 1, grad_X.size());
    vDSP_vadd(grad_X.grad_ptr(), 1, dX_ln1.resolve_read_ptr(), 1, grad_X.grad_ptr(), 1, grad_X.size());
}

// ============================================================================
// Multi-Layer TitanModel Implementation
// ============================================================================

TitanModel::TitanModel(
    size_t vocab_size,
    size_t d_model,
    size_t num_layers,
    size_t n_heads,
    size_t max_seq_len,
    size_t d_ff
) : vocab_size(vocab_size),
    d_model(d_model),
    num_layers(num_layers),
    n_heads(n_heads),
    max_seq_len(max_seq_len) {

    token_embeddings = Tensor(vocab_size, d_model);
    pos_embeddings   = Tensor(max_seq_len, d_model);

    layers.reserve(num_layers);
    for (size_t l = 0; l < num_layers; ++l) {
        layers.emplace_back(d_model, n_heads, max_seq_len, d_ff);
    }

    final_ln = LayerNorm(d_model);
    lm_head  = Tensor(d_model, vocab_size);

    init_weights(42);
}

void TitanModel::init_weights(uint64_t seed) {
    token_embeddings.init_xavier(vocab_size, d_model, seed + 1);
    pos_embeddings.init_xavier(max_seq_len, d_model, seed + 2);

    for (size_t l = 0; l < num_layers; ++l) {
        layers[l].init_weights(seed + 100 * (l + 1));
    }

    final_ln.init_weights();
    lm_head.init_xavier(d_model, vocab_size, seed + 999);

    zero_gradients();
}

void TitanModel::zero_gradients() noexcept {
    token_embeddings.zero_gradients();
    pos_embeddings.zero_gradients();

    for (auto& layer : layers) {
        layer.zero_gradients();
    }

    final_ln.zero_gradients();
    lm_head.zero_gradients();
}

void TitanModel::clear_cache() noexcept {
    layer_inputs.clear();
    cached_final_norm = Tensor();
    cached_input_tokens.clear();

    for (auto& layer : layers) {
        layer.clear_cache();
    }
    final_ln.clear_cache();
}

size_t TitanModel::parameter_count() const noexcept {
    size_t total = token_embeddings.size() + pos_embeddings.size();
    for (const auto& layer : layers) {
        total += layer.parameter_count();
    }
    total += final_ln.parameter_count();
    total += lm_head.size();
    return total;
}

Tensor TitanModel::forward(const std::vector<uint32_t>& tokens) {
    const size_t T = tokens.size();
    if (T == 0) {
        throw std::invalid_argument("Cannot forward pass empty token sequence");
    }
    if (T > max_seq_len) {
        throw std::invalid_argument(
            "Sequence length (" + std::to_string(T) +
            ") exceeds max_seq_len (" + std::to_string(max_seq_len) + ")"
        );
    }

    cached_input_tokens = tokens;
    layer_inputs.clear();
    layer_inputs.reserve(num_layers + 1);

    // 1. Embedding lookup & Positional addition: X = TokEmb + PosEmb
    Tensor X(T, d_model, 0.0f);
    for (size_t t = 0; t < T; ++t) {
        uint32_t tok = tokens[t];
        if (tok >= vocab_size) {
            throw std::out_of_range("Token ID " + std::to_string(tok) + " exceeds vocab_size");
        }

        const float* tok_src = token_embeddings.data_ptr() + tok * d_model;
        const float* pos_src = pos_embeddings.data_ptr() + t * d_model;
        float* dst = X.data_ptr() + t * d_model;

        vDSP_vadd(tok_src, 1, pos_src, 1, dst, 1, d_model);
    }

    layer_inputs.push_back(X);

    // 2. Sequential Transformer Blocks Traversal
    Tensor current = X;
    for (size_t l = 0; l < num_layers; ++l) {
        current = layers[l].forward(current);
        layer_inputs.push_back(current);
    }

    // 3. Final Layer Normalization
    cached_final_norm = final_ln.forward(current);

    // 4. LM Head Projection: Logits = cached_final_norm x lm_head (T x vocab_size)
    return Tensor::matmul(cached_final_norm, lm_head);
}

void TitanModel::backward(const Tensor& grad_logits) {
    const size_t T = cached_input_tokens.size();
    if (grad_logits.rows != T || grad_logits.cols != vocab_size) {
        throw std::invalid_argument("grad_logits shape mismatch in TitanModel::backward");
    }

    // 1. LM Head Backward: Logits = cached_final_norm x lm_head
    // d(cached_final_norm) = grad_logits x lm_head^T
    const float* d_logits_ptr = grad_logits.resolve_read_ptr();

    Tensor d_final_norm(T, d_model, 0.0f);
    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasTrans,
        static_cast<int>(T), static_cast<int>(d_model), static_cast<int>(vocab_size),
        1.0f, d_logits_ptr, static_cast<int>(vocab_size),
        lm_head.data_ptr(), static_cast<int>(vocab_size),
        0.0f, d_final_norm.data_ptr(), static_cast<int>(d_model)
    );

    // d(lm_head) += cached_final_norm^T x grad_logits
    cblas_sgemm(
        CblasRowMajor, CblasTrans, CblasNoTrans,
        static_cast<int>(d_model), static_cast<int>(vocab_size), static_cast<int>(T),
        1.0f, cached_final_norm.data_ptr(), static_cast<int>(d_model),
        d_logits_ptr, static_cast<int>(vocab_size),
        1.0f, lm_head.grad_ptr(), static_cast<int>(vocab_size)
    );

    // 2. Final Layer Normalization Backward
    Tensor d_layers_out(T, d_model, 0.0f);
    final_ln.backward(d_final_norm, d_layers_out);

    // 3. Reverse Transformer Blocks Traversal
    Tensor current_grad = d_layers_out;
    for (int l = static_cast<int>(num_layers) - 1; l >= 0; --l) {
        Tensor next_grad(T, d_model, 0.0f);
        layers[l].backward(current_grad, next_grad);
        current_grad = std::move(next_grad);
    }

    // 4. Token & Positional Embeddings Backward
    const float* dx_base = current_grad.resolve_read_ptr();
    for (size_t t = 0; t < T; ++t) {
        uint32_t tok = cached_input_tokens[t];
        const float* dx = dx_base + t * d_model;

        float* tok_grad = token_embeddings.grad_ptr() + tok * d_model;
        float* pos_grad = pos_embeddings.grad_ptr() + t * d_model;

        vDSP_vadd(tok_grad, 1, dx, 1, tok_grad, 1, d_model);
        vDSP_vadd(pos_grad, 1, dx, 1, pos_grad, 1, d_model);
    }
}

std::vector<ParameterRef> TitanModel::get_parameters() {
    std::vector<ParameterRef> params;
    params.push_back({&token_embeddings, "token_embeddings", true});
    params.push_back({&pos_embeddings, "pos_embeddings", true});

    for (size_t l = 0; l < layers.size(); ++l) {
        std::string pfx = "layer_" + std::to_string(l) + ".";
        params.push_back({&layers[l].ln_1.gamma, pfx + "ln_1.gamma", false});
        params.push_back({&layers[l].ln_1.beta, pfx + "ln_1.beta", false});

        params.push_back({&layers[l].attn.W_q, pfx + "attn.W_q", true});
        params.push_back({&layers[l].attn.b_q, pfx + "attn.b_q", false});
        params.push_back({&layers[l].attn.W_k, pfx + "attn.W_k", true});
        params.push_back({&layers[l].attn.b_k, pfx + "attn.b_k", false});
        params.push_back({&layers[l].attn.W_v, pfx + "attn.W_v", true});
        params.push_back({&layers[l].attn.b_v, pfx + "attn.b_v", false});
        params.push_back({&layers[l].attn.W_o, pfx + "attn.W_o", true});
        params.push_back({&layers[l].attn.b_o, pfx + "attn.b_o", false});

        params.push_back({&layers[l].ln_2.gamma, pfx + "ln_2.gamma", false});
        params.push_back({&layers[l].ln_2.beta, pfx + "ln_2.beta", false});

        params.push_back({&layers[l].ffn.W_1, pfx + "ffn.W_1", true});
        params.push_back({&layers[l].ffn.b_1, pfx + "ffn.b_1", false});
        params.push_back({&layers[l].ffn.W_2, pfx + "ffn.W_2", true});
        params.push_back({&layers[l].ffn.b_2, pfx + "ffn.b_2", false});
    }

    params.push_back({&final_ln.gamma, "final_ln.gamma", false});
    params.push_back({&final_ln.beta, "final_ln.beta", false});
    params.push_back({&lm_head, "lm_head", true});

    return params;
}

namespace {
struct alignas(8) CheckpointHeader {
    uint32_t magic{0x54495457}; // 'TITW'
    uint32_t version{1};
    uint64_t total_params{0};
    uint32_t vocab_size{0};
    uint32_t d_model{0};
    uint32_t num_layers{0};
    uint32_t n_heads{0};
    uint32_t max_seq_len{0};
    uint32_t step{0};
    float loss{0.0f};
    uint32_t reserved{0};
};
} // anonymous namespace

bool TitanModel::save_weights(const std::string& filepath, uint32_t step, float loss) const {
    std::ofstream out(filepath, std::ios::binary);
    if (!out.is_open()) return false;

    CheckpointHeader header{
        .magic = 0x54495457,
        .version = 1,
        .total_params = static_cast<uint64_t>(parameter_count()),
        .vocab_size = static_cast<uint32_t>(vocab_size),
        .d_model = static_cast<uint32_t>(d_model),
        .num_layers = static_cast<uint32_t>(num_layers),
        .n_heads = static_cast<uint32_t>(n_heads),
        .max_seq_len = static_cast<uint32_t>(max_seq_len),
        .step = step,
        .loss = loss,
        .reserved = 0
    };

    out.write(reinterpret_cast<const char*>(&header), sizeof(header));

    auto params = const_cast<TitanModel*>(this)->get_parameters();
    for (const auto& p : params) {
        out.write(
            reinterpret_cast<const char*>(p.tensor->data.data()),
            static_cast<std::streamsize>(p.tensor->bytes())
        );
    }

    return out.good();
}

bool TitanModel::load_weights(const std::string& filepath) {
    std::ifstream in(filepath, std::ios::binary);
    if (!in.is_open()) return false;

    CheckpointHeader header{};
    in.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!in.good() || header.magic != 0x54495457) return false;

    if (header.vocab_size != vocab_size || header.d_model != d_model ||
        header.num_layers != num_layers || header.n_heads != n_heads) {
        return false;
    }

    auto params = get_parameters();
    for (const auto& p : params) {
        in.read(
            reinterpret_cast<char*>(p.tensor->data.data()),
            static_cast<std::streamsize>(p.tensor->bytes())
        );
        if (!in.good()) return false;
    }

    return true;
}

std::string TitanModel::generate(
    const std::string& prompt,
    const Tokenizer& tokenizer,
    size_t max_new_tokens,
    float temperature,
    uint64_t seed,
    bool stream_to_stdout
) {
    std::vector<TokenId> tokens = tokenizer.encode(prompt, true, false);
    if (tokens.empty()) tokens.push_back(BOS_TOKEN_ID);

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    std::string generated_text;

    for (size_t step = 0; step < max_new_tokens; ++step) {
        std::vector<uint32_t> ctx_tokens;
        if (tokens.size() > max_seq_len) {
            ctx_tokens.assign(tokens.end() - max_seq_len, tokens.end());
        } else {
            ctx_tokens = tokens;
        }

        Tensor logits = forward(ctx_tokens);
        const size_t T = ctx_tokens.size();
        const float* next_logits = logits.data_ptr() + (T - 1) * vocab_size;

        TokenId next_token = 0;
        if (temperature <= 1e-4f) {
            float best_logit = next_logits[0];
            for (size_t c = 1; c < vocab_size; ++c) {
                if (next_logits[c] > best_logit) {
                    best_logit = next_logits[c];
                    next_token = static_cast<TokenId>(c);
                }
            }
        } else {
            std::vector<float> scaled_logits(vocab_size);
            float max_val = -1e9f;
            for (size_t c = 0; c < vocab_size; ++c) {
                float val = next_logits[c] / temperature;
                scaled_logits[c] = val;
                if (val > max_val) max_val = val;
            }

            float sum_exp = 0.0f;
            for (size_t c = 0; c < vocab_size; ++c) {
                scaled_logits[c] = std::exp(scaled_logits[c] - max_val);
                sum_exp += scaled_logits[c];
            }

            float r_sum = 1.0f / (sum_exp > 0.0f ? sum_exp : 1e-12f);
            for (size_t c = 0; c < vocab_size; ++c) {
                scaled_logits[c] *= r_sum;
            }

            float r = dist(rng);
            float cum_prob = 0.0f;
            for (size_t c = 0; c < vocab_size; ++c) {
                cum_prob += scaled_logits[c];
                if (r <= cum_prob || c + 1 == vocab_size) {
                    next_token = static_cast<TokenId>(c);
                    break;
                }
            }
        }

        if (next_token == EOS_TOKEN_ID) {
            break;
        }

        tokens.push_back(next_token);
        std::string piece = tokenizer.token_to_piece(next_token);
        generated_text.append(piece);

        if (stream_to_stdout) {
            std::cout << piece << std::flush;
        }
    }

    return generated_text;
}

} // namespace titan
