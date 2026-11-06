#include "trainer.hpp"

#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>
#include <chrono>
#include <stdexcept>
#include <thread>
#include <Accelerate/Accelerate.h>

namespace titan {

// ============================================================================
// AdamW Optimizer Implementation
// ============================================================================

AdamW::AdamW(
    const std::vector<ParameterRef>& parameters,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay
) : lr(lr),
    beta1(beta1),
    beta2(beta2),
    eps(eps),
    weight_decay(weight_decay),
    params(parameters) {

    states.reserve(params.size());
    for (const auto& pref : params) {
        OptimizerMomentState state;
        const size_t n = pref.tensor->size();
        state.m.assign(n, 0.0f);
        state.v.assign(n, 0.0f);
        states.push_back(std::move(state));
    }
}

void AdamW::zero_grad() noexcept {
    for (auto& pref : params) {
        pref.tensor->zero_gradients();
    }
}

void AdamW::step() {
    step_count++;

    // Calculate bias correction factors
    const float alpha_t = 1.0f - std::pow(beta1, static_cast<float>(step_count));
    const float beta_t  = 1.0f - std::pow(beta2, static_cast<float>(step_count));
    const float step_factor = lr * (std::sqrt(beta_t) / (alpha_t > 0.0f ? alpha_t : 1e-12f));
    const float one_minus_beta1 = 1.0f - beta1;
    const float one_minus_beta2 = 1.0f - beta2;
    const float wd_factor = lr * weight_decay;

    // Vectorized update pass across contiguous tensor data layouts
    for (size_t p = 0; p < params.size(); ++p) {
        ParameterRef& pref = params[p];
        Tensor* tensor = pref.tensor;

        float* W = tensor->data_ptr();
        const float* G = tensor->grad_ptr();
        float* M = states[p].m.data();
        float* V = states[p].v.data();
        const size_t N = tensor->size();
        const bool apply_decay = pref.weight_decay && (weight_decay > 0.0f);

        // Apple Silicon Neon vectorizes this contiguous loop directly
        for (size_t i = 0; i < N; ++i) {
            float g = G[i];

            // 1. Decoupled Weight Decay: theta = theta - lr * lambda * theta
            if (apply_decay) {
                W[i] -= wd_factor * W[i];
            }

            // 2. First Moment: m_t = beta1 * m_{t-1} + (1 - beta1) * g
            M[i] = beta1 * M[i] + one_minus_beta1 * g;

            // 3. Second Moment: v_t = beta2 * v_{t-1} + (1 - beta2) * g^2
            V[i] = beta2 * V[i] + one_minus_beta2 * (g * g);

            // 4. Parameter Update with Bias-Corrected Step
            W[i] -= step_factor * (M[i] / (std::sqrt(V[i]) + eps));
        }
    }
}

// ============================================================================
// Trainer Implementation
// ============================================================================

Trainer::Trainer(
    TitanModel& model,
    const Tokenizer& tokenizer,
    const TrainingConfig& config
) : model(model),
    tokenizer(tokenizer),
    config(config),
    optimizer(
        model.get_parameters(),
        config.learning_rate,
        config.beta1,
        config.beta2,
        config.eps,
        config.weight_decay
    ),
    governor(config.governor) {}

float Trainer::compute_scheduled_lr(size_t step) const noexcept {
    if (step < config.warmup_steps) {
        return config.learning_rate * static_cast<float>(step + 1) /
               static_cast<float>(config.warmup_steps > 0 ? config.warmup_steps : 1);
    }
    if (step >= config.max_steps) {
        return config.min_learning_rate;
    }

    // Cosine decay schedule
    const float decay_ratio = static_cast<float>(step - config.warmup_steps) /
                              static_cast<float>(config.max_steps - config.warmup_steps);
    const float coeff = 0.5f * (1.0f + std::cos(3.14159265358979323846f * decay_ratio));
    return config.min_learning_rate + coeff * (config.learning_rate - config.min_learning_rate);
}

void Trainer::train() {
    std::cout << "====================================================================\n";
    std::cout << "  TITAN-LM Training Engine: Starting AdamW Optimization Loop\n";
    std::cout << "====================================================================\n";

    // 1. Memory-map pre-tokenized binary dataset
    MappedTokenBuffer dataset = tokenizer.mmap_tokens_binary(config.dataset_path);
    if (!dataset.is_valid() || dataset.size() <= config.seq_len + 1) {
        throw std::runtime_error(
            "Dataset in " + config.dataset_path + " contains insufficient tokens (" +
            std::to_string(dataset.size()) + ") for sequence length " + std::to_string(config.seq_len)
        );
    }

    const size_t num_tokens = dataset.size();
    const size_t record_stride = 2 * config.seq_len;
    const size_t total_sft_records = (record_stride > 0) ? (num_tokens / record_stride) : 0;
    const size_t max_sample_offset = (num_tokens > config.seq_len + 1) ? (num_tokens - (config.seq_len + 1)) : 0;

    std::cout << "  Dataset File: " << config.dataset_path << " (" << num_tokens << " mapped tokens)\n";
    if (config.is_sft) {
        std::cout << "  Training Mode: Supervised Fine-Tuning (SFT) with Target Loss Masking\n";
        std::cout << "  Aligned Records: " << total_sft_records << " complete dialogue turns (len=" << config.seq_len << ")\n";
    } else {
        std::cout << "  Training Mode: Autoregressive Next-Token Pre-Training\n";
    }
    std::cout << "  Model Parameters: " << model.parameter_count() << " parameters\n";
    std::cout << "  Context Length: " << config.seq_len << " tokens | Micro-Batch: " << config.batch_size << "\n";
    std::cout << "  Optimization Steps: " << config.max_steps << " | Warmup: " << config.warmup_steps << " steps\n";
    std::cout << "  Peak LR: " << config.learning_rate << " | Min LR: " << config.min_learning_rate << "\n";
    if (config.start_step > 1) {
        std::cout << "  Resuming Training from Step: " << config.start_step << " of " << config.max_steps << "\n";
    }
    if (config.governor.enabled) {
        auto init_tel = governor.sample();
        std::cout << "  Smart Governor: ACTIVE [Profile: " << config.governor.profile_name << "]\n";
        std::cout << "  Hardware Sensors: " << init_tel.sensor_count << " Apple Silicon die sensors detected"
                  << " | Docked Setup: " << (init_tel.is_docked ? "YES" : "NO") << "\n";
        std::cout << "  Setpoints: Target ≤ " << std::fixed << std::setprecision(0) << config.governor.target_temp_c
                  << "°C (Full Throttle) | Emergency > " << config.governor.emergency_temp_c
                  << "°C | Resume ≤ " << config.governor.resume_temp_c << "°C\n";
        std::cout << "  Initial State: " << std::fixed << std::setprecision(1) << init_tel.effective_temp_c 
                  << "°C (Peak: " << init_tel.max_temp_c << "°C, Avg: " << init_tel.avg_temp_c << "°C) | "
                  << "Power: " << (init_tel.on_ac ? "AC Connected" : "Battery") 
                  << " (" << init_tel.battery_pct << "%" << (init_tel.is_charging ? ", Charging" : "") << ")\n";
    }
    std::cout << "--------------------------------------------------------------------\n";

    std::mt19937_64 rng(config.seed);
    std::uniform_int_distribution<size_t> offset_dist(0, max_sample_offset);
    std::uniform_int_distribution<size_t> record_dist(0, total_sft_records > 0 ? (total_sft_records - 1) : 0);

    auto start_time = std::chrono::high_resolution_clock::now();
    size_t total_tokens_processed = 0;

    for (size_t step = config.start_step; step <= config.max_steps; ++step) {
        auto step_start = std::chrono::high_resolution_clock::now();

        // 2. Set cosine-decayed scheduled learning rate
        const float current_lr = compute_scheduled_lr(step);
        optimizer.set_lr(current_lr);

        // 3. Reset gradients for upcoming batch accumulation
        model.zero_gradients();

        float total_batch_loss = 0.0f;

        // 4. Gradient Accumulation Loop across Micro-Batches
        for (size_t b = 0; b < config.batch_size; ++b) {
            std::vector<uint32_t> X_tokens;
            std::vector<uint32_t> Y_targets;

            if (config.is_sft && total_sft_records > 0) {
                // SFT Mode: Sample a complete, turn-aligned conversation record
                const size_t rec_idx = record_dist(rng);
                const size_t rec_offset = rec_idx * record_stride;

                X_tokens.assign(
                    dataset.data_ptr() + rec_offset,
                    dataset.data_ptr() + rec_offset + config.seq_len
                );
                // Y_targets has prompt & padding tokens masked with IGNORE_INDEX
                Y_targets.assign(
                    dataset.data_ptr() + rec_offset + config.seq_len,
                    dataset.data_ptr() + rec_offset + record_stride
                );
            } else {
                // Pre-training Mode: Continuous next-token prediction stream
                const size_t offset = offset_dist(rng);

                X_tokens.assign(
                    dataset.data_ptr() + offset,
                    dataset.data_ptr() + offset + config.seq_len
                );
                Y_targets.assign(
                    dataset.data_ptr() + offset + 1,
                    dataset.data_ptr() + offset + 1 + config.seq_len
                );
            }

            // Forward Pass
            Tensor logits = model.forward(X_tokens);

            // Compute Cross-Entropy Loss & Analytical Logits Gradient
            Tensor grad_logits(config.seq_len, model.vocab_size, 0.0f);
            float loss = Tensor::cross_entropy_loss(logits, Y_targets, grad_logits);
            total_batch_loss += loss;

            // Scale gradient for batch accumulation
            if (config.batch_size > 1) {
                float scale_factor = 1.0f / static_cast<float>(config.batch_size);
                vDSP_vsmul(
                    grad_logits.grad_ptr(), 1,
                    &scale_factor,
                    grad_logits.grad_ptr(), 1,
                    grad_logits.size()
                );
            }

            // Backward Autograd Pass: propagates through LM head, blocks, and embeddings
            model.backward(grad_logits);

            // Dynamic closed-loop pacing hook: inserts micro-delay if die temp is elevated
            governor.apply_inter_batch_pacing();
        }

        const float avg_step_loss = total_batch_loss / static_cast<float>(config.batch_size);

        // 5. Optimizer Update Pass
        optimizer.step();

        total_tokens_processed += config.batch_size * config.seq_len;

        auto step_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> step_dur = step_end - step_start;
        double step_tokens_per_sec = (config.batch_size * config.seq_len) / (step_dur.count() * 1e-3);

        // 6. Routine Telemetry Logging
        if (step % config.log_interval == 0 || step == 1 || step == config.max_steps) {
            std::cout << "  Step [" << std::setw(4) << step << "/" << config.max_steps << "] | "
                      << "Loss: " << std::fixed << std::setprecision(4) << avg_step_loss << " | "
                      << "LR: " << std::scientific << std::setprecision(2) << current_lr << " | "
                      << "Speed: " << std::fixed << std::setprecision(1) << step_tokens_per_sec << " tok/s";
            if (config.governor.enabled) {
                const auto& tel = governor.last_telemetry;
                std::cout << " | " << std::fixed << std::setprecision(1) << tel.effective_temp_c << "°C [" << tel.thermal_zone << "]";
                if (tel.active_pacing_ms > 0) {
                    std::cout << " (Pace: " << tel.active_pacing_ms << "ms)";
                }
            }
            std::cout << "\n" << std::flush;
        }

        // 7. Checkpoint Serialization
        if (step % config.checkpoint_interval == 0 || step == config.max_steps) {
            if (model.save_weights(config.checkpoint_path, static_cast<uint32_t>(step), avg_step_loss)) {
                std::cout << "  --> Checkpoint saved: " << config.checkpoint_path
                          << " (Step " << step << " | Loss: " << std::fixed << std::setprecision(4)
                          << avg_step_loss << ")\n"
                          << std::flush;
            }
        }

        // 8. Smart Thermal & Battery Governor: Closed-loop real-time telemetry check
        if (config.governor.enabled && step < config.max_steps) {
            governor.on_step_completed(step, config.max_steps, avg_step_loss, model, config.checkpoint_path);
        }
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_duration = total_end - start_time;
    double overall_tok_per_sec = total_tokens_processed / total_duration.count();

    std::cout << "--------------------------------------------------------------------\n";
    std::cout << "  Training Complete: " << config.max_steps << " steps in "
              << std::fixed << std::setprecision(2) << total_duration.count() << " seconds ("
              << overall_tok_per_sec << " overall tokens/sec)\n";
    std::cout << "====================================================================\n\n";
}

} // namespace titan

