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
    float effective_temp_c{0.0f};    // Weighted control temperature (60% avg + 40% peak)
    int sensor_count{0};             // Number of active IOKit thermal sensors
    int battery_pct{100};            // Current battery charge (0-100)
    bool on_ac{true};                // True if drawing from AC adapter (MagSafe/USB-C)
    bool is_charging{false};         // True if battery is actively charging
    bool is_docked{false};           // True if docked / external display detected
    uint32_t active_pacing_ms{0};    // Current adaptive inter-batch duty cycle delay (ms)
    std::string thermal_zone;        // "OPTIMAL", "ELEVATED", "WARM", "HOT", "EMERGENCY"
};

// ============================================================================
// Thermal Operating Profiles
// ============================================================================
enum class ThermalProfile {
    AUTO,    // Auto-detects DOCKED if external display/dock connected or on AC, else MOBILE
    MOBILE,  // Laptop on lap: conservative setpoints (target 76°C, emergency 88°C, resume 65°C)
    DOCKED   // Docked/Desk with cooling pad: high-performance setpoints (target 86°C, emergency 95°C, resume 78°C)
};

// ============================================================================
// Smart Adaptive Closed-Loop Governor Configuration
// ============================================================================
struct GovernorConfig {
    bool enabled{true};
    ThermalProfile profile{ThermalProfile::AUTO};
    std::string profile_name{"AUTO"};

    // Continuous Closed-Loop Temperature Setpoints (°C)
    float target_temp_c{86.0f};      // Below this: Green zone (0ms pacing, 100% full throttle)
    float throttle_temp_c{90.0f};    // Adaptive micro-pacing (5ms-35ms)
    float alert_temp_c{93.0f};       // Steeper cooling pacing (40ms-85ms)
    float emergency_temp_c{95.0f};   // Red zone: instant checkpoint save + sensor cooldown loop
    float resume_temp_c{78.0f};      // Dynamic cooldown floor: resumes immediately once temp <= floor

    // Intelligent Battery Management
    int min_battery_percent{20};     // Safe reserve: pause if on battery and <= 20%
    bool battery_saver_on_dc{true};  // When on battery, apply 20ms eco-pacing to protect battery health
    uint32_t dc_eco_pacing_ms{20};   // Base eco-pacing delay on DC power

    void apply_profile(ThermalProfile p);
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

    // Hardware setup detection
    static bool is_docked_setup();

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

