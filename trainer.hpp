#pragma once
#ifndef TITAN_TRAINER_HPP
#define TITAN_TRAINER_HPP

#include "model.hpp"
#include "tokenizer.hpp"
#include <vector>
#include <string>
#include <cstddef>
#include <cstdint>
#include <chrono>

namespace titan {

// ============================================================================
// Training Engine Configuration
// ============================================================================
struct TrainingConfig {
    size_t batch_size{4};           // Gradient accumulation batch size (B)
    size_t seq_len{32};             // Context length (T)
    size_t max_steps{100};          // Total optimization steps
    size_t warmup_steps{10};        // Linear warmup steps
    float learning_rate{1e-3f};     // Peak learning rate
    float min_learning_rate{1e-4f}; // Cosine decayed minimum learning rate
    float weight_decay{0.01f};      // Decoupled weight decay (lambda)
    float beta1{0.9f};              // First moment decay factor
    float beta2{0.95f};             // Second moment decay factor
    float eps{1e-8f};               // Numerical stability denominator epsilon
    size_t log_interval{10};        // Frequency of telemetry reporting
    size_t checkpoint_interval{25}; // Frequency of saving titan_weights.bin
    std::string checkpoint_path{"titan_weights.bin"};
    std::string dataset_path{"titan_tokens.bin"};
    uint64_t seed{1337};
    bool is_sft{false};             // Supervised Fine-Tuning mode with paired (X, Y_masked) records
    size_t start_step{1};           // Starting step (for seamless checkpoint resumption)

    // ── Smart Thermal & Battery Governor (Duty-Cycle Training) ─────────────
    bool smart_governor_enabled{true};
    size_t work_window_minutes{30};       // Continuous training duration before cooling pause (default 30m)
    size_t cooldown_minutes{10};          // In-process cooling pause duration (default 10m)
    int min_battery_percent{20};          // Pause if battery drops below 20% on DC
};

// ============================================================================
// AdamW Optimizer State & Core Engine
// ============================================================================
struct OptimizerMomentState {
    std::vector<float> m; // First moment vector (mean)
    std::vector<float> v; // Second moment vector (uncentered variance)
};

class AdamW {
public:
    float lr{1e-3f};
    float beta1{0.9f};
    float beta2{0.95f};
    float eps{1e-8f};
    float weight_decay{0.01f};
    uint64_t step_count{0};

    std::vector<ParameterRef> params;
    std::vector<OptimizerMomentState> states;

    // --- Lifecycle ---
    AdamW() = default;
    AdamW(
        const std::vector<ParameterRef>& parameters,
        float lr = 1e-3f,
        float beta1 = 0.9f,
        float beta2 = 0.95f,
        float eps = 1e-8f,
        float weight_decay = 0.01f
    );

    // Vectorized optimization step across contiguous memory
    void step();

    // Reset parameter gradients across all registered tensors
    void zero_grad() noexcept;

    void set_lr(float new_lr) noexcept { lr = new_lr; }
};

// ============================================================================
// Training Orchestrator Engine
// ============================================================================
class Trainer {
public:
    TitanModel& model;
    const Tokenizer& tokenizer;
    TrainingConfig config;
    AdamW optimizer;

    Trainer(TitanModel& model, const Tokenizer& tokenizer, const TrainingConfig& config);

    // Executes the complete training loop:
    // Memory maps dataset, samples (B, T) batches, computes cross-entropy loss,
    // runs backward autograd pass, steps AdamW with cosine LR decay, and saves checkpoints.
    void train();

    // Computes scheduled learning rate with linear warmup and cosine decay
    [[nodiscard]] float compute_scheduled_lr(size_t step) const noexcept;
};

} // namespace titan

#endif // TITAN_TRAINER_HPP

