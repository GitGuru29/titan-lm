#pragma once
#ifndef TITAN_GOVERNOR_HPP
#define TITAN_GOVERNOR_HPP

#include <string>
#include <cstdint>
#include <chrono>

namespace titan {

// Forward declarations
class TitanModel;

// ============================================================================
// Real-Time Hardware Sensor Telemetry
// ============================================================================
struct GovernorTelemetry {
    float max_temp_c{0.0f};          // Peak die sensor temperature (°C)
    float avg_temp_c{0.0f};          // Average active die sensor temperature (°C)
    int sensor_count{0};             // Number of active IOKit thermal sensors
    int battery_pct{100};            // Current battery charge (0-100)
    bool on_ac{true};                // True if drawing from AC adapter (MagSafe/USB-C)
    bool is_charging{false};         // True if battery is actively charging
    uint32_t active_pacing_ms{0};     // Current adaptive inter-batch duty cycle delay (ms)
    std::string thermal_zone;        // "NOMINAL", "ELEVATED", "HOT", "EMERGENCY"
};

// ============================================================================
// Smart Adaptive Closed-Loop Governor Configuration
// ============================================================================
struct GovernorConfig {
    bool enabled{true};

    // Continuous Closed-Loop Temperature Setpoints (°C)
    float target_temp_c{76.0f};      // Below this: Green zone (0ms pacing, 100% full throttle)
    float throttle_temp_c{82.0f};    // 76°C-82°C: Yellow zone (smooth 5ms-35ms adaptive micro-pacing)
    float alert_temp_c{86.0f};       // 82°C-86°C: Orange zone (40ms-80ms cooling pacing)
    float emergency_temp_c{88.0f};   // > 88°C: Red zone (instant checkpoint save + active sensor cooldown)
    float resume_temp_c{65.0f};      // Dynamic cooldown floor: resumes immediately once temp <= 65°C

    // Intelligent Battery Management
    int min_battery_percent{20};     // Safe reserve: pause if on battery and <= 20%
    bool battery_saver_on_dc{true};  // When on battery, apply 20ms eco-pacing to protect battery health
    uint32_t dc_eco_pacing_ms{20};   // Base eco-pacing delay on DC power
};

// ============================================================================
// Hardware Governor Engine (IOKit HID & Power Management Core)
// ============================================================================
class HardwareGovernor {
public:
    GovernorConfig config;
    GovernorTelemetry last_telemetry;

    HardwareGovernor(const GovernorConfig& cfg = {});
    ~HardwareGovernor();

    // Query hardware sensors in < 2ms without subprocess overhead
    [[nodiscard]] GovernorTelemetry sample();

    // Inter-batch micro-pacing hook: stabilizes die temperatures continuously
    // without ever stopping the training loop
    void apply_inter_batch_pacing();

    // Post-step closed-loop check:
    // Calculates proportional feedback, updates pacing, and triggers smart dynamic
    // cooldown if emergency threshold is breached (auto-resumes as soon as temp drops)
    void on_step_completed(
        size_t step,
        size_t max_steps,
        float current_loss,
        TitanModel& model,
        const std::string& checkpoint_path
    );

private:
    void* hid_client{nullptr};
    void* hid_matching_dict{nullptr};
    std::chrono::high_resolution_clock::time_point last_sample_time;
    uint32_t current_pacing_ms{0};

    void init_hid_client();
    void release_hid_client();
    void execute_emergency_cooldown(
        size_t step,
        float current_loss,
        TitanModel& model,
        const std::string& checkpoint_path
    );
};

} // namespace titan

#endif // TITAN_GOVERNOR_HPP
