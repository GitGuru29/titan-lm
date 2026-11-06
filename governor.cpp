#include "governor.hpp"
#include "model.hpp"

#include <iostream>
#include <iomanip>
#include <thread>
#include <chrono>
#include <cmath>
#include <algorithm>

#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>
#include <IOKit/ps/IOPowerSources.h>
#include <IOKit/ps/IOPSKeys.h>

extern "C" {
    typedef void* IOHIDEventSystemClientRef;
    typedef void* IOHIDServiceClientRef;
    typedef void* IOHIDEventRef;

    IOHIDEventSystemClientRef IOHIDEventSystemClientCreate(CFAllocatorRef allocator);
    void IOHIDEventSystemClientSetMatching(IOHIDEventSystemClientRef client, CFDictionaryRef matching);
    CFArrayRef IOHIDEventSystemClientCopyServices(IOHIDEventSystemClientRef client);
    CFTypeRef IOHIDServiceClientCopyProperty(IOHIDServiceClientRef service, CFStringRef key);
    IOHIDEventRef IOHIDServiceClientCopyEvent(IOHIDServiceClientRef service, int64_t type, int32_t options, int64_t param);
    double IOHIDEventGetFloatValue(IOHIDEventRef event, uint32_t field);
}

namespace titan {

HardwareGovernor::HardwareGovernor(const GovernorConfig& cfg)
    : config(cfg), last_sample_time(std::chrono::high_resolution_clock::now()) {
    init_hid_client();
    (void)sample();
}

HardwareGovernor::~HardwareGovernor() {
    release_hid_client();
}

void HardwareGovernor::init_hid_client() {
    IOHIDEventSystemClientRef client = IOHIDEventSystemClientCreate(kCFAllocatorDefault);
    if (!client) {
        hid_client = nullptr;
        return;
    }
    hid_client = static_cast<void*>(client);

    // PrimaryUsagePage: 0xff00 (Apple Vendor), PrimaryUsage: 5 (Thermal Sensors)
    int page = 0xff00;
    int usage = 5;
    CFNumberRef page_ref = CFNumberCreate(kCFAllocatorDefault, kCFNumberIntType, &page);
    CFNumberRef usage_ref = CFNumberCreate(kCFAllocatorDefault, kCFNumberIntType, &usage);
    CFStringRef key1 = CFSTR("PrimaryUsagePage");
    CFStringRef key2 = CFSTR("PrimaryUsage");

    const void* keys[2] = { key1, key2 };
    const void* vals[2] = { page_ref, usage_ref };
    CFDictionaryRef matching = CFDictionaryCreate(
        kCFAllocatorDefault, keys, vals, 2,
        &kCFTypeDictionaryKeyCallBacks,
        &kCFTypeDictionaryValueCallBacks
    );

    IOHIDEventSystemClientSetMatching(client, matching);
    hid_matching_dict = static_cast<void*>(const_cast<__CFDictionary*>(matching));

    CFRelease(page_ref);
    CFRelease(usage_ref);
}

void HardwareGovernor::release_hid_client() {
    if (hid_matching_dict) {
        CFRelease(static_cast<CFDictionaryRef>(hid_matching_dict));
        hid_matching_dict = nullptr;
    }
    if (hid_client) {
        CFRelease(static_cast<IOHIDEventSystemClientRef>(hid_client));
        hid_client = nullptr;
    }
}

GovernorTelemetry HardwareGovernor::sample() {
    GovernorTelemetry tel{};
    tel.max_temp_c = 0.0f;
    tel.avg_temp_c = 0.0f;
    tel.sensor_count = 0;
    tel.battery_pct = 100;
    tel.on_ac = true;
    tel.is_charging = false;

    // ── 1. Query Real Hardware Thermal Sensors via IOKit HID ────────────────
    if (hid_client) {
        IOHIDEventSystemClientRef client = static_cast<IOHIDEventSystemClientRef>(hid_client);
        CFArrayRef services = IOHIDEventSystemClientCopyServices(client);
        if (services) {
            CFIndex count = CFArrayGetCount(services);
            float sum_temp = 0.0f;
            int valid_sensors = 0;

            for (CFIndex i = 0; i < count; ++i) {
                IOHIDServiceClientRef s = (IOHIDServiceClientRef)CFArrayGetValueAtIndex(services, i);
                // IOHIDEventTypeTemperature = 15
                IOHIDEventRef event = IOHIDServiceClientCopyEvent(s, 15, 0, 0);
                if (event) {
                    // Field 15 << 16 = 0x000F0000
                    double val = IOHIDEventGetFloatValue(event, 15 << 16);
                    if (val > 15.0 && val < 125.0) {
                        float temp = static_cast<float>(val);
                        if (temp > tel.max_temp_c) {
                            tel.max_temp_c = temp;
                        }
                        sum_temp += temp;
                        valid_sensors++;
                    }
                    CFRelease(event);
                }
            }
            tel.sensor_count = valid_sensors;
            if (valid_sensors > 0) {
                tel.avg_temp_c = sum_temp / static_cast<float>(valid_sensors);
            }
            CFRelease(services);
        }
    }

    // Fallback if HID thermal sensors unavailable
    if (tel.sensor_count == 0) {
        tel.max_temp_c = 50.0f;
        tel.avg_temp_c = 45.0f;
    }

    // ── 2. Query Battery & Power Status via IOPowerSources ──────────────────
    CFTypeRef ps_info = IOPSCopyPowerSourcesInfo();
    if (ps_info) {
        CFArrayRef ps_list = IOPSCopyPowerSourcesList(ps_info);
        if (ps_list) {
            CFIndex count = CFArrayGetCount(ps_list);
            for (CFIndex i = 0; i < count; ++i) {
                CFTypeRef ps = CFArrayGetValueAtIndex(ps_list, i);
                CFDictionaryRef desc = IOPSGetPowerSourceDescription(ps_info, ps);
                if (!desc) continue;

                CFStringRef state_ref = (CFStringRef)CFDictionaryGetValue(desc, CFSTR(kIOPSPowerSourceStateKey));
                if (state_ref) {
                    tel.on_ac = (CFStringCompare(state_ref, CFSTR(kIOPSACPowerValue), 0) == kCFCompareEqualTo);
                }

                CFNumberRef cap_ref = (CFNumberRef)CFDictionaryGetValue(desc, CFSTR(kIOPSCurrentCapacityKey));
                if (cap_ref) {
                    int cap = 100;
                    CFNumberGetValue(cap_ref, kCFNumberIntType, &cap);
                    tel.battery_pct = cap;
                }

                CFBooleanRef chg_ref = (CFBooleanRef)CFDictionaryGetValue(desc, CFSTR(kIOPSIsChargingKey));
                if (chg_ref) {
                    tel.is_charging = CFBooleanGetValue(chg_ref);
                }
            }
            CFRelease(ps_list);
        }
        CFRelease(ps_info);
    }

    // ── 3. Proportional Closed-Loop Pacing Feedback Calculation ─────────────
    uint32_t pacing = 0;

    if (!tel.on_ac && config.battery_saver_on_dc) {
        // Base eco-pacing on battery power to prevent high current draw
        pacing += config.dc_eco_pacing_ms;
    }

    if (tel.max_temp_c <= config.target_temp_c) {
        // Green zone (Cold / Optimal): run at full hardware speed
        tel.thermal_zone = "OPTIMAL";
    } else if (tel.max_temp_c <= config.throttle_temp_c) {
        // Yellow zone (Elevated): proportional micro-delay (5ms - 35ms)
        tel.thermal_zone = "ELEVATED";
        float ratio = (tel.max_temp_c - config.target_temp_c) / (config.throttle_temp_c - config.target_temp_c);
        pacing += static_cast<uint32_t>(std::lerp(5.0f, 35.0f, ratio));
    } else if (tel.max_temp_c <= config.alert_temp_c) {
        // Orange zone (Warm): steeper pacing to stabilize junction temperature
        tel.thermal_zone = "WARM";
        float ratio = (tel.max_temp_c - config.throttle_temp_c) / (config.alert_temp_c - config.throttle_temp_c);
        pacing += static_cast<uint32_t>(std::lerp(40.0f, 85.0f, ratio));
    } else {
        // Red zone (Hot): strong back-off
        tel.thermal_zone = "HOT";
        pacing += 120;
    }

    tel.active_pacing_ms = pacing;
    current_pacing_ms = pacing;
    last_telemetry = tel;
    last_sample_time = std::chrono::high_resolution_clock::now();

    return tel;
}

void HardwareGovernor::apply_inter_batch_pacing() {
    if (!config.enabled || current_pacing_ms == 0) return;
    std::this_thread::sleep_for(std::chrono::milliseconds(current_pacing_ms));
}

void HardwareGovernor::on_step_completed(
    size_t step,
    [[maybe_unused]] size_t max_steps,
    float current_loss,
    TitanModel& model,
    const std::string& checkpoint_path
) {
    if (!config.enabled) return;

    // Sample hardware sensors every step (< 1ms overhead)
    GovernorTelemetry tel = sample();

    // ── Emergency Thermal Brake: Temperature exceeded emergency threshold ───
    if (tel.max_temp_c >= config.emergency_temp_c) {
        execute_emergency_cooldown(step, current_loss, model, checkpoint_path);
        return;
    }

    // ── Battery Protection: Safe Reserve Floor ──────────────────────────────
    if (!tel.on_ac && tel.battery_pct <= config.min_battery_percent) {
        std::cout << "\n====================================================================\n";
        std::cout << "  [Smart Governor] Battery Critical (" << tel.battery_pct << "% on Battery Power)\n";
        std::cout << "  Safeguarding parameters: saving checkpoint to " << checkpoint_path << "...\n";
        model.save_weights(checkpoint_path, static_cast<uint32_t>(step), current_loss);
        std::cout << "  --> Checkpoint safely synchronized: " << checkpoint_path << "\n";
        std::cout << "  Training paused until MagSafe/USB-C charger is connected...\n";
        std::cout << "====================================================================\n" << std::flush;

        while (true) {
            std::this_thread::sleep_for(std::chrono::seconds(5));
            tel = sample();
            if (tel.on_ac || tel.battery_pct > config.min_battery_percent) {
                std::cout << "  [Smart Governor] AC Power detected! Battery: " << tel.battery_pct 
                          << "% (Charging: " << (tel.is_charging ? "Yes" : "No") << "). Resuming training!\n" << std::flush;
                break;
            }
        }
    }
}

void HardwareGovernor::execute_emergency_cooldown(
    size_t step,
    float current_loss,
    TitanModel& model,
    const std::string& checkpoint_path
) {
    std::cout << "\n====================================================================\n";
    std::cout << "  [Smart Governor] Thermal Emergency Alert! Die Temp: " 
              << std::fixed << std::setprecision(1) << last_telemetry.max_temp_c << "°C (Ceiling: " 
              << config.emergency_temp_c << "°C)\n";
    std::cout << "  Saving interim checkpoint to disk before thermal pause...\n";
    model.save_weights(checkpoint_path, static_cast<uint32_t>(step), current_loss);
    std::cout << "  --> Checkpoint safely synchronized: " << checkpoint_path << "\n";
    std::cout << "  Entering Dynamic Sensor Cooldown (polling every 500ms)...\n";
    std::cout << "  (Will resume instantly the moment die temp drops to " << config.resume_temp_c << "°C)\n";
    std::cout << "====================================================================\n" << std::flush;

    auto cooldown_start = std::chrono::high_resolution_clock::now();
    int report_tick = 0;

    // Dynamic Sensor Poll Loop — resumes as soon as hardware actually cools down!
    while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        GovernorTelemetry tel = sample();

        if (report_tick % 10 == 0) { // Report every 5 seconds
            auto now = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = now - cooldown_start;
            std::cout << "  [Smart Governor] Cooling: Peak " << std::fixed << std::setprecision(1) 
                      << tel.max_temp_c << "°C (Target: ≤" << config.resume_temp_c 
                      << "°C) | Elapsed: " << std::fixed << std::setprecision(1) << elapsed.count() 
                      << "s | Fan/Pad cooling active...\n" << std::flush;
        }
        report_tick++;

        // Break dynamically as soon as temperature reaches safe floor
        if (tel.max_temp_c <= config.resume_temp_c) {
            auto cooldown_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> total_cooldown = cooldown_end - cooldown_start;

            std::cout << "\n====================================================================\n";
            std::cout << "  [Smart Governor] Safe Thermal Floor Reached: " 
                      << std::fixed << std::setprecision(1) << tel.max_temp_c << "°C in " 
                      << std::fixed << std::setprecision(1) << total_cooldown.count() << " seconds!\n";
            std::cout << "  CPU Die Cooled Down. Resuming training at full speed!\n";
            std::cout << "====================================================================\n\n" << std::flush;
            break;
        }
    }
}

} // namespace titan
