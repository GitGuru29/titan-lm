#pragma once
#ifndef TITAN_TENSOR_HPP
#define TITAN_TENSOR_HPP

#include <vector>
#include <span>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <initializer_list>
#include <stdexcept>
#include <Accelerate/Accelerate.h>

namespace titan {

// ============================================================================
// Tensor: 2D High-Performance Contiguous Autograd Tensor for Apple Silicon M4
// ============================================================================
struct Tensor {
    size_t rows{0};
    size_t cols{0};

    // Primary activations / parameters (contiguous flat array for L1/L2 cache locality)
    std::vector<float> data;

    // Paired backpropagation gradient buffer (identical contiguous flat footprint)
    std::vector<float> grad;

    // --- Constructors & Lifecycle ---
    Tensor() noexcept = default;
    Tensor(size_t r, size_t c, float fill_value = 0.0f);
    Tensor(size_t r, size_t c, const std::vector<float>& initial_data);
    Tensor(size_t r, size_t c, std::initializer_list<float> initial_data);

    Tensor(const Tensor&) = default;
    Tensor& operator=(const Tensor&) = default;
    Tensor(Tensor&&) noexcept = default;
    Tensor& operator=(Tensor&&) noexcept = default;
    ~Tensor() = default;

    // Shape & Dimension properties
    [[nodiscard]] size_t size() const noexcept { return rows * cols; }
    [[nodiscard]] bool empty() const noexcept { return rows == 0 || cols == 0; }
    [[nodiscard]] size_t bytes() const noexcept { return size() * sizeof(float); }

    // Direct pointer access for Apple Accelerate / BLAS bindings
    [[nodiscard]] float* data_ptr() noexcept { return data.data(); }
    [[nodiscard]] const float* data_ptr() const noexcept { return data.data(); }
    [[nodiscard]] float* grad_ptr() noexcept { return grad.data(); }
    [[nodiscard]] const float* grad_ptr() const noexcept { return grad.data(); }
    [[nodiscard]] const float* resolve_read_ptr() const noexcept {
        for (float g : grad) {
            if (g != 0.0f) return grad.data();
        }
        return data.data();
    }

    // C++20 spans for zero-overhead safe view slicing
    [[nodiscard]] std::span<float> data_span() noexcept { return {data.data(), data.size()}; }
    [[nodiscard]] std::span<const float> data_span() const noexcept { return {data.data(), data.size()}; }
    [[nodiscard]] std::span<float> grad_span() noexcept { return {grad.data(), grad.size()}; }
    [[nodiscard]] std::span<const float> grad_span() const noexcept { return {grad.data(), grad.size()}; }

    // 2D element-wise access
    [[nodiscard]] float& operator()(size_t r, size_t c) noexcept {
        return data[r * cols + c];
    }
    [[nodiscard]] const float& operator()(size_t r, size_t c) const noexcept {
        return data[r * cols + c];
    }

    [[nodiscard]] float& grad_at(size_t r, size_t c) noexcept {
        return grad[r * cols + c];
    }
    [[nodiscard]] const float& grad_at(size_t r, size_t c) const noexcept {
        return grad[r * cols + c];
    }

    [[nodiscard]] float& at(size_t r, size_t c);
    [[nodiscard]] const float& at(size_t r, size_t c) const;

    // --- State & Memory Initialization ---
    void zero_data() noexcept;
    void zero_gradients() noexcept;
    void fill_constant(float value) noexcept;
    void fill_random(float min_val, float max_val, uint64_t seed = 42);
    void init_xavier(size_t fan_in, size_t fan_out, uint64_t seed = 42);
    void init_kaiming(size_t fan_in, uint64_t seed = 42);

    // Reshapes 2D dimensions preserving total element count
    void reshape(size_t new_rows, size_t new_cols);

    // In-place gradient accumulation (grad += incoming)
    void accumulate_grad(const Tensor& incoming);
    void accumulate_grad(std::span<const float> incoming_grad);

    // --- Apple Matrix Coprocessor (AMX) Hardware Operations ---

    // Forward Matrix Multiplication: C = A x B via Apple native cblas_sgemm
    // Shape constraints: A (M x K), B (K x N) -> C (M x N)
    [[nodiscard]] static Tensor matmul(const Tensor& A, const Tensor& B);

    // Backward Matrix Multiplication: accumulates gradients into grad_A and grad_B
    // dA += dC x B^T  (M x N) x (N x K) -> (M x K)
    // dB += A^T x dC  (K x M) x (M x N) -> (K x N)
    static void backward_matmul(
        const Tensor& A,
        const Tensor& B,
        const Tensor& grad_C,
        Tensor& grad_A,
        Tensor& grad_B
    );

    // Member convenience wrapper
    [[nodiscard]] Tensor matmul(const Tensor& B) const { return matmul(*this, B); }

    // --- Element-Wise & Bias Layer Operations ---

    // Element-wise addition: C = A + B (A and B must have identical shapes)
    [[nodiscard]] static Tensor add(const Tensor& A, const Tensor& B);
    static void backward_add(const Tensor& grad_C, Tensor& grad_A, Tensor& grad_B);

    // Bias Addition: Y = X + bias, broadcasting row vector bias (1 x N) across all rows of X (M x N)
    [[nodiscard]] static Tensor add_bias(const Tensor& X, const Tensor& bias);
    static void backward_add_bias(const Tensor& grad_Y, Tensor& grad_X, Tensor& grad_bias);

    // Member convenience methods
    [[nodiscard]] Tensor forward_add(const Tensor& other) const { return add(*this, other); }
    [[nodiscard]] Tensor forward_add_bias(const Tensor& bias) const { return add_bias(*this, bias); }

    // --- Activation Functions & Autograd Graph Operations ---

    // ReLU Activation: Y = max(0, X)
    [[nodiscard]] static Tensor forward_relu(const Tensor& input);
    static void backward_relu(const Tensor& input, const Tensor& grad_output, Tensor& grad_input);

    // Member ReLU wrappers
    [[nodiscard]] Tensor forward_relu() const { return forward_relu(*this); }
    void backward_relu(const Tensor& grad_output);

    // Numerically-stable Row-wise Softmax:
    // Y(r, c) = exp(X(r, c) - max_r) / sum_c exp(X(r, c) - max_r)
    [[nodiscard]] static Tensor forward_softmax(const Tensor& input);

    // Backward Softmax:
    // dX(r, c) = Y(r, c) * (dY(r, c) - sum_k dY(r, k) * Y(r, k))
    static void backward_softmax(
        const Tensor& output_probs,
        const Tensor& grad_output,
        Tensor& grad_input
    );

    // Member Softmax wrappers
    [[nodiscard]] Tensor forward_softmax() const { return forward_softmax(*this); }
    void backward_softmax(const Tensor& output_probs, const Tensor& grad_output);

    // --- Loss & Optimization Utilities ---

    // Cross-Entropy Loss computation for language modeling
    // Given logits (B x V) and target integer tokens (B):
    // Computes mean cross-entropy loss and accumulates gradients into grad_logits
    static float cross_entropy_loss(
        const Tensor& logits,
        const std::vector<uint32_t>& targets,
        Tensor& grad_logits
    );

    // Diagnostic utilities
    void print_shape(std::string_view name = "Tensor") const;
    void print_data(std::string_view name = "Tensor", size_t max_rows = 4, size_t max_cols = 6) const;
    void print_grad(std::string_view name = "Tensor", size_t max_rows = 4, size_t max_cols = 6) const;
};

} // namespace titan

#endif // TITAN_TENSOR_HPP

