#include "tensor.hpp"

#include <cmath>
#include <random>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cassert>

namespace titan {

namespace {

// Helper to resolve incoming gradient buffer from either .grad or .data
inline const float* resolve_grad_ptr(const Tensor& t) noexcept {
    for (float g : t.grad) {
        if (g != 0.0f) {
            return t.grad.data();
        }
    }
    return t.data.data();
}

} // anonymous namespace

// ============================================================================
// Constructors & Element Access
// ============================================================================

Tensor::Tensor(size_t r, size_t c, float fill_value)
    : rows(r), cols(c), data(r * c, fill_value), grad(r * c, 0.0f) {}

Tensor::Tensor(size_t r, size_t c, const std::vector<float>& initial_data)
    : rows(r), cols(c), data(initial_data), grad(r * c, 0.0f) {
    if (data.size() != r * c) {
        throw std::invalid_argument(
            "Initial data size (" + std::to_string(data.size()) +
            ") does not match dimensions (" + std::to_string(r) + "x" + std::to_string(c) + ")"
        );
    }
}

Tensor::Tensor(size_t r, size_t c, std::initializer_list<float> initial_data)
    : rows(r), cols(c), data(initial_data), grad(r * c, 0.0f) {
    if (data.size() != r * c) {
        throw std::invalid_argument(
            "Initializer list size (" + std::to_string(data.size()) +
            ") does not match dimensions (" + std::to_string(r) + "x" + std::to_string(c) + ")"
        );
    }
}

float& Tensor::at(size_t r, size_t c) {
    if (r >= rows || c >= cols) {
        throw std::out_of_range(
            "Tensor index out of bounds: (" + std::to_string(r) + ", " + std::to_string(c) +
            ") for shape (" + std::to_string(rows) + ", " + std::to_string(cols) + ")"
        );
    }
    return data[r * cols + c];
}

const float& Tensor::at(size_t r, size_t c) const {
    if (r >= rows || c >= cols) {
        throw std::out_of_range(
            "Tensor index out of bounds: (" + std::to_string(r) + ", " + std::to_string(c) +
            ") for shape (" + std::to_string(rows) + ", " + std::to_string(cols) + ")"
        );
    }
    return data[r * cols + c];
}

// ============================================================================
// Memory & State Management
// ============================================================================

void Tensor::zero_data() noexcept {
    std::fill(data.begin(), data.end(), 0.0f);
}

void Tensor::zero_gradients() noexcept {
    std::fill(grad.begin(), grad.end(), 0.0f);
}

void Tensor::fill_constant(float value) noexcept {
    std::fill(data.begin(), data.end(), value);
}

void Tensor::fill_random(float min_val, float max_val, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(min_val, max_val);
    for (float& val : data) {
        val = dist(rng);
    }
}

void Tensor::init_xavier(size_t fan_in, size_t fan_out, uint64_t seed) {
    float limit = std::sqrt(6.0f / static_cast<float>(fan_in + fan_out));
    fill_random(-limit, limit, seed);
}

void Tensor::init_kaiming(size_t fan_in, uint64_t seed) {
    float std_dev = std::sqrt(2.0f / static_cast<float>(fan_in));
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> dist(0.0f, std_dev);
    for (float& val : data) {
        val = dist(rng);
    }
}

void Tensor::reshape(size_t new_rows, size_t new_cols) {
    if (new_rows * new_cols != rows * cols) {
        throw std::invalid_argument(
            "Cannot reshape tensor of size " + std::to_string(rows * cols) +
            " to (" + std::to_string(new_rows) + "x" + std::to_string(new_cols) + ")"
        );
    }
    rows = new_rows;
    cols = new_cols;
}

void Tensor::accumulate_grad(const Tensor& incoming) {
    if (incoming.size() != size()) {
        throw std::invalid_argument("Size mismatch in accumulate_grad");
    }
    const float* src = resolve_grad_ptr(incoming);
    vDSP_vadd(grad.data(), 1, src, 1, grad.data(), 1, size());
}

void Tensor::accumulate_grad(std::span<const float> incoming_grad) {
    if (incoming_grad.size() != size()) {
        throw std::invalid_argument("Size mismatch in accumulate_grad span");
    }
    vDSP_vadd(grad.data(), 1, incoming_grad.data(), 1, grad.data(), 1, size());
}

// ============================================================================
// Apple Matrix Coprocessor (AMX) Hardware Bindings via Accelerate cblas_sgemm
// ============================================================================

Tensor Tensor::matmul(const Tensor& A, const Tensor& B) {
    if (A.cols != B.rows) {
        throw std::invalid_argument(
            "Incompatible shapes for matmul: (" + std::to_string(A.rows) + "x" + std::to_string(A.cols) +
            ") and (" + std::to_string(B.rows) + "x" + std::to_string(B.cols) + ")"
        );
    }

    Tensor C(A.rows, B.cols, 0.0f);

    // Call Apple's native AMX-accelerated GEMM: C = 1.0 * A * B + 0.0 * C
    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        static_cast<int>(A.rows),
        static_cast<int>(B.cols),
        static_cast<int>(A.cols),
        1.0f,
        A.data.data(),
        static_cast<int>(A.cols),
        B.data.data(),
        static_cast<int>(B.cols),
        0.0f,
        C.data.data(),
        static_cast<int>(B.cols)
    );

    return C;
}

void Tensor::backward_matmul(
    const Tensor& A,
    const Tensor& B,
    const Tensor& grad_C,
    Tensor& grad_A,
    Tensor& grad_B
) {
    if (A.cols != B.rows || grad_C.rows != A.rows || grad_C.cols != B.cols) {
        throw std::invalid_argument("Dimension mismatch in backward_matmul");
    }
    if (grad_A.rows != A.rows || grad_A.cols != A.cols) {
        throw std::invalid_argument("grad_A shape mismatch in backward_matmul");
    }
    if (grad_B.rows != B.rows || grad_B.cols != B.cols) {
        throw std::invalid_argument("grad_B shape mismatch in backward_matmul");
    }

    const float* dC_ptr = resolve_grad_ptr(grad_C);

    // 1. dA += dC * B^T
    // dC: (M x N), B: (K x N) -> B^T: (N x K) -> dA: (M x K)
    // Beta = 1.0f accumulates directly into grad_A.grad
    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasTrans,
        static_cast<int>(A.rows),
        static_cast<int>(A.cols),
        static_cast<int>(B.cols),
        1.0f,
        dC_ptr,
        static_cast<int>(B.cols),
        B.data.data(),
        static_cast<int>(B.cols),
        1.0f,
        grad_A.grad.data(),
        static_cast<int>(A.cols)
    );

    // 2. dB += A^T * dC
    // A: (M x K) -> A^T: (K x M), dC: (M x N) -> dB: (K x N)
    // Beta = 1.0f accumulates directly into grad_B.grad
    cblas_sgemm(
        CblasRowMajor,
        CblasTrans,
        CblasNoTrans,
        static_cast<int>(A.cols),
        static_cast<int>(B.cols),
        static_cast<int>(A.rows),
        1.0f,
        A.data.data(),
        static_cast<int>(A.cols),
        dC_ptr,
        static_cast<int>(B.cols),
        1.0f,
        grad_B.grad.data(),
        static_cast<int>(B.cols)
    );
}

// ============================================================================
// Element-Wise Operations & Bias Layers
// ============================================================================

Tensor Tensor::add(const Tensor& A, const Tensor& B) {
    if (A.rows != B.rows || A.cols != B.cols) {
        throw std::invalid_argument(
            "Shape mismatch in Tensor::add: (" + std::to_string(A.rows) + "x" + std::to_string(A.cols) +
            ") vs (" + std::to_string(B.rows) + "x" + std::to_string(B.cols) + ")"
        );
    }

    Tensor C(A.rows, A.cols);
    vDSP_vadd(A.data.data(), 1, B.data.data(), 1, C.data.data(), 1, A.size());
    return C;
}

void Tensor::backward_add(const Tensor& grad_C, Tensor& grad_A, Tensor& grad_B) {
    if (grad_A.size() != grad_C.size() || grad_B.size() != grad_C.size()) {
        throw std::invalid_argument("Size mismatch in Tensor::backward_add");
    }

    const float* dC_ptr = resolve_grad_ptr(grad_C);
    vDSP_vadd(grad_A.grad.data(), 1, dC_ptr, 1, grad_A.grad.data(), 1, grad_A.size());
    vDSP_vadd(grad_B.grad.data(), 1, dC_ptr, 1, grad_B.grad.data(), 1, grad_B.size());
}

Tensor Tensor::add_bias(const Tensor& X, const Tensor& bias) {
    if (bias.size() != X.cols) {
        throw std::invalid_argument(
            "Bias vector size (" + std::to_string(bias.size()) +
            ") must match tensor column dimension (" + std::to_string(X.cols) + ")"
        );
    }

    Tensor Y(X.rows, X.cols);
    const float* bias_ptr = bias.data.data();

    // Broadcast bias row vector across each row of X using SIMD vDSP_vadd
    for (size_t r = 0; r < X.rows; ++r) {
        vDSP_vadd(
            X.data.data() + r * X.cols, 1,
            bias_ptr, 1,
            Y.data.data() + r * X.cols, 1,
            X.cols
        );
    }

    return Y;
}

void Tensor::backward_add_bias(const Tensor& grad_Y, Tensor& grad_X, Tensor& grad_bias) {
    if (grad_X.size() != grad_Y.size()) {
        throw std::invalid_argument("grad_X size mismatch in backward_add_bias");
    }
    if (grad_bias.size() != grad_X.cols) {
        throw std::invalid_argument("grad_bias size mismatch in backward_add_bias");
    }

    const float* dY_ptr = resolve_grad_ptr(grad_Y);

    // 1. grad_X += grad_Y
    vDSP_vadd(grad_X.grad.data(), 1, dY_ptr, 1, grad_X.grad.data(), 1, grad_X.size());

    // 2. Accumulate gradients across batch/sequence rows into bias gradient
    for (size_t r = 0; r < grad_X.rows; ++r) {
        vDSP_vadd(
            grad_bias.grad.data(), 1,
            dY_ptr + r * grad_X.cols, 1,
            grad_bias.grad.data(), 1,
            grad_X.cols
        );
    }
}

// ============================================================================
// Activation Functions & Autograd Graph Derivatives
// ============================================================================

Tensor Tensor::forward_relu(const Tensor& input) {
    Tensor Y(input.rows, input.cols);
    const size_t n = input.size();
    const float* src = input.data.data();
    float* dst = Y.data.data();

    // Apple Silicon Neon vectorizes this loop cleanly to fmax instructions
    for (size_t i = 0; i < n; ++i) {
        dst[i] = (src[i] > 0.0f) ? src[i] : 0.0f;
    }
    return Y;
}

void Tensor::backward_relu(const Tensor& input, const Tensor& grad_output, Tensor& grad_input) {
    if (input.size() != grad_output.size() || input.size() != grad_input.size()) {
        throw std::invalid_argument("Size mismatch in backward_relu");
    }

    const size_t n = input.size();
    const float* src = input.data.data();
    const float* dy = resolve_grad_ptr(grad_output);
    float* dx = grad_input.grad.data();

    for (size_t i = 0; i < n; ++i) {
        if (src[i] > 0.0f) {
            dx[i] += dy[i];
        }
    }
}

void Tensor::backward_relu(const Tensor& grad_output) {
    backward_relu(*this, grad_output, *this);
}

Tensor Tensor::forward_softmax(const Tensor& input) {
    Tensor Y(input.rows, input.cols);

    for (size_t r = 0; r < input.rows; ++r) {
        const float* x_row = input.data.data() + r * input.cols;
        float* y_row = Y.data.data() + r * input.cols;

        // Numerical stability: subtract row maximum
        float max_val = 0.0f;
        vDSP_maxv(x_row, 1, &max_val, input.cols);

        // Exponentiate
        for (size_t c = 0; c < input.cols; ++c) {
            y_row[c] = std::exp(x_row[c] - max_val);
        }

        // Sum exponentiated values
        float sum_val = 0.0f;
        vDSP_sve(y_row, 1, &sum_val, input.cols);

        // Normalize probabilities
        float inv_sum = 1.0f / (sum_val > 0.0f ? sum_val : 1e-12f);
        vDSP_vsmul(y_row, 1, &inv_sum, y_row, 1, input.cols);
    }

    return Y;
}

void Tensor::backward_softmax(
    const Tensor& output_probs,
    const Tensor& grad_output,
    Tensor& grad_input
) {
    if (output_probs.rows != grad_output.rows || output_probs.cols != grad_output.cols) {
        throw std::invalid_argument("Shape mismatch in backward_softmax");
    }
    if (output_probs.rows != grad_input.rows || output_probs.cols != grad_input.cols) {
        throw std::invalid_argument("grad_input shape mismatch in backward_softmax");
    }

    const size_t rows = output_probs.rows;
    const size_t cols = output_probs.cols;
    const float* dy_ptr = resolve_grad_ptr(grad_output);

    for (size_t r = 0; r < rows; ++r) {
        const float* y_row = output_probs.data.data() + r * cols;
        const float* dy_row = dy_ptr + r * cols;
        float* dx_row = grad_input.grad.data() + r * cols;

        // dot = sum_k (dY_k * Y_k)
        float dot = 0.0f;
        vDSP_dotpr(y_row, 1, dy_row, 1, &dot, cols);

        // dX_c += Y_c * (dY_c - dot)
        for (size_t c = 0; c < cols; ++c) {
            dx_row[c] += y_row[c] * (dy_row[c] - dot);
        }
    }
}

void Tensor::backward_softmax(const Tensor& output_probs, const Tensor& grad_output) {
    backward_softmax(output_probs, grad_output, *this);
}

// ============================================================================
// Loss & Autograd Utilities
// ============================================================================

float Tensor::cross_entropy_loss(
    const Tensor& logits,
    const std::vector<uint32_t>& targets,
    Tensor& grad_logits
) {
    if (logits.rows != targets.size()) {
        throw std::invalid_argument("Logits row count must match targets count");
    }
    if (grad_logits.rows != logits.rows || grad_logits.cols != logits.cols) {
        throw std::invalid_argument("grad_logits shape mismatch in cross_entropy_loss");
    }

    const size_t B = logits.rows;
    const size_t V = logits.cols;
    Tensor probs = forward_softmax(logits);

    float total_loss = 0.0f;
    const float inv_batch = 1.0f / static_cast<float>(B);

    for (size_t b = 0; b < B; ++b) {
        uint32_t target = targets[b];
        if (target >= V) {
            throw std::out_of_range("Target token ID exceeds vocabulary dimension");
        }

        float p = probs(b, target);
        total_loss += -std::log(std::max(p, 1e-12f));

        // Softmax + Cross Entropy combined analytical gradient: (P_c - 1{c == target}) / B
        for (size_t c = 0; c < V; ++c) {
            float indicator = (c == target) ? 1.0f : 0.0f;
            grad_logits.grad_at(b, c) += (probs(b, c) - indicator) * inv_batch;
        }
    }

    return total_loss * inv_batch;
}

// ============================================================================
// Diagnostic Telemetry
// ============================================================================

void Tensor::print_shape(std::string_view name) const {
    std::cout << "[" << name << "] Shape: (" << rows << " x " << cols << ") ["
              << size() << " elements, " << (bytes() / 1024.0f) << " KB]\n";
}

void Tensor::print_data(std::string_view name, size_t max_rows, size_t max_cols) const {
    std::cout << "[" << name << " Data (" << rows << "x" << cols << ")]:\n";
    size_t r_limit = std::min(rows, max_rows);
    size_t c_limit = std::min(cols, max_cols);

    for (size_t r = 0; r < r_limit; ++r) {
        std::cout << "  [";
        for (size_t c = 0; c < c_limit; ++c) {
            std::cout << std::fixed << std::setprecision(4) << data[r * cols + c]
                      << (c + 1 < c_limit ? ", " : "");
        }
        if (cols > max_cols) std::cout << ", ...";
        std::cout << "]\n";
    }
    if (rows > max_rows) std::cout << "  ...\n";
}

void Tensor::print_grad(std::string_view name, size_t max_rows, size_t max_cols) const {
    std::cout << "[" << name << " Grad (" << rows << "x" << cols << ")]:\n";
    size_t r_limit = std::min(rows, max_rows);
    size_t c_limit = std::min(cols, max_cols);

    for (size_t r = 0; r < r_limit; ++r) {
        std::cout << "  [";
        for (size_t c = 0; c < c_limit; ++c) {
            std::cout << std::fixed << std::setprecision(4) << grad[r * cols + c]
                      << (c + 1 < c_limit ? ", " : "");
        }
        if (cols > max_cols) std::cout << ", ...";
        std::cout << "]\n";
    }
    if (rows > max_rows) std::cout << "  ...\n";
}

} // namespace titan

