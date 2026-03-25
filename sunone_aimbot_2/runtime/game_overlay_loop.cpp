#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cwchar>
#include <deque>
#include <filesystem>
#include <mutex>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "capture.h"
#include "Game_overlay.h"
#include "mouse.h"
#include "other_tools.h"
#include "runtime/thread_loops.h"
#include "sunone_aimbot_2.h"
#include "aim_kalman.h"

#ifdef USE_CUDA
#include "depth/depth_anything_trt.h"
#include "depth/depth_mask.h"
#include "tensorrt/nvinf.h"
#endif

extern std::string g_iconLastError;

namespace
{
std::string g_lastIconPath;
int g_iconImageId = 0;
std::mutex g_iconMutex;
}
static void draw_target_correction_demo_game_overlay(Game_overlay* overlay, float centerX, float centerY)
{
    if (!overlay)
        return;

    const float scale = 4.0f;
    float near_px = config.nearRadius * scale;
    float snap_px = config.snapRadius * scale;
    near_px = std::max(10.0f, near_px);
    snap_px = std::max(6.0f, std::min(snap_px, near_px - 4.0f));

    overlay->AddCircle({ centerX, centerY, near_px }, ARGB(180, 80, 120, 255), 2.0f);
    overlay->AddCircle({ centerX, centerY, snap_px }, ARGB(180, 255, 100, 100), 2.0f);

    static float dist_px = 0.0f;
    static float vel_px = 0.0f;
    static auto last_t = std::chrono::steady_clock::now();

    auto now = std::chrono::steady_clock::now();
    double dt = std::chrono::duration<double>(now - last_t).count();
    last_t = now;
    dt = std::max(0.0, std::min(dt, 0.1));

    if (dist_px <= 0.0f || dist_px > near_px)
        dist_px = near_px;

    double dist_units = dist_px / scale;
    double speed_mult;
    if (dist_units < config.snapRadius)
    {
        speed_mult = config.minSpeedMultiplier * config.snapBoostFactor;
    }
    else if (dist_units < config.nearRadius)
    {
        double t = dist_units / config.nearRadius;
        double crv = 1.0 - std::pow(1.0 - t, config.speedCurveExponent);
        speed_mult = config.minSpeedMultiplier +
            (config.maxSpeedMultiplier - config.minSpeedMultiplier) * crv;
    }
    else
    {
        double norm = std::max(0.0, std::min(dist_units / config.nearRadius, 1.0));
        speed_mult = config.minSpeedMultiplier +
            (config.maxSpeedMultiplier - config.minSpeedMultiplier) * norm;
    }

    float max_multiplier = std::max(0.1f, config.maxSpeedMultiplier);
    float demo_duration_s = std::max(0.6f, std::min(2.2f / max_multiplier, 3.0f));
    float base_px_s = near_px / demo_duration_s;
    vel_px = base_px_s * static_cast<float>(speed_mult);
    dist_px -= vel_px * static_cast<float>(dt);
    if (dist_px <= 0.0f)
        dist_px = near_px;

    overlay->FillCircle({ centerX - dist_px, centerY, 4.0f }, ARGB(255, 255, 255, 80));
}

struct AimSimVec2
{
    double x = 0.0;
    double y = 0.0;
};

struct AimSimHistorySample
{
    double timeSec = 0.0;
    AimSimVec2 target;
    AimSimVec2 aim;
};

struct AimSimQueuedMove
{
    double executeAtSec = 0.0;
    int mx = 0;
    int my = 0;
};

struct AimSimulationState
{
    bool initialized = false;
    bool controllerInitialized = false;
    int panelW = 0;
    int panelH = 0;

    std::mt19937 rng{ std::random_device{}() };
    std::chrono::steady_clock::time_point lastWallTime{};

    double simTimeSec = 0.0;
    double accumulatorSec = 0.0;
    double currentFps = 100.0;
    double targetFps = 100.0;
    double fpsRetargetSec = 0.0;
    double targetMotionRetargetSec = 0.0;
    double lastFrameDtSec = 1.0 / 100.0;

    AimSimVec2 targetPos;
    AimSimVec2 targetVel;
    AimSimVec2 targetDesiredVel;
    AimSimVec2 aimPos;
    AimSimVec2 observedTarget;
    AimSimVec2 predictedTarget;

    double usedCaptureDelayMs = 0.0;
    double usedInferenceDelayMs = 0.0;
    double usedInputDelayMs = 0.0;
    double usedExtraDelayMs = 0.0;
    double usedTotalDelayMs = 0.0;
    double lastErrorPx = 0.0;
    int lastMoveX = 0;
    int lastMoveY = 0;

    // Controller state matched to current MouseThread::moveMousePivot + Kalman logic.
    aim::AimKalman2D ctrlKalman;
    aim::AimKalmanTelemetry ctrlKalmanTelemetry;
    bool kalmanSettingsInitialized = false;
    aim::AimKalmanSettings kalmanSettingsSnapshot;
    double ctrlPrevTimeSec = 0.0;
    double kalmanLookaheadSec = 0.0;
    double kalmanInnovationPx = 0.0;
    AimSimVec2 kalmanMeasuredTarget;
    AimSimVec2 kalmanEstimatedTarget;
    AimSimVec2 kalmanVelocity;

    // Keep WindMouse state between frames so simulation doesn't "reset" behavior.
    double windCarryX = 0.0;
    double windCarryY = 0.0;
    double windVelX = 0.0;
    double windVelY = 0.0;
    double windNoiseX = 0.0;
    double windNoiseY = 0.0;
    double windFracX = 0.0;
    double windFracY = 0.0;
    double windPatternX = 0.0;
    double windPatternY = 0.0;
    double windPatternPhaseA = 0.0;
    double windPatternPhaseB = 0.0;
    double windPatternRateA = 0.0;
    double windPatternRateB = 0.0;
    std::mt19937 windRng{ std::random_device{}() };
    bool windEnabledSnapshot = false;
    double windGSnapshot = 0.0;
    double windWSnapshot = 0.0;
    double windMSnapshot = 0.0;
    double windDSnapshot = 0.0;

    std::deque<AimSimHistorySample> history;
    std::deque<AimSimQueuedMove> queuedMoves;
    std::deque<AimSimVec2> targetTrail;
    std::deque<AimSimVec2> aimTrail;
};

static double aim_sim_random_range(AimSimulationState& s, double minV, double maxV)
{
    std::uniform_real_distribution<double> dist(minV, maxV);
    return dist(s.rng);
}

static double aim_sim_vec_len(const AimSimVec2& v)
{
    return std::hypot(v.x, v.y);
}

static aim::AimKalmanSettings aim_sim_build_kalman_settings_from_config()
{
    aim::AimKalmanSettings settings;
    settings.enabled = config.kalman_enabled;
    settings.process_noise_position = static_cast<double>(config.kalman_process_noise_position);
    settings.process_noise_velocity = static_cast<double>(config.kalman_process_noise_velocity);
    settings.measurement_noise = static_cast<double>(config.kalman_measurement_noise);
    settings.velocity_damping = static_cast<double>(config.kalman_velocity_damping);
    settings.max_velocity = static_cast<double>(config.kalman_max_velocity);
    settings.warmup_frames = config.kalman_warmup_frames;
    return settings;
}

static bool aim_sim_kalman_settings_equal(const aim::AimKalmanSettings& a, const aim::AimKalmanSettings& b)
{
    return
        (a.enabled == b.enabled) &&
        (std::abs(a.process_noise_position - b.process_noise_position) <= 1e-6) &&
        (std::abs(a.process_noise_velocity - b.process_noise_velocity) <= 1e-6) &&
        (std::abs(a.measurement_noise - b.measurement_noise) <= 1e-6) &&
        (std::abs(a.velocity_damping - b.velocity_damping) <= 1e-6) &&
        (std::abs(a.max_velocity - b.max_velocity) <= 1e-6) &&
        (a.warmup_frames == b.warmup_frames);
}

static void aim_sim_sync_kalman_config(AimSimulationState& s)
{
    const aim::AimKalmanSettings settings = aim_sim_build_kalman_settings_from_config();
    const bool changed = !s.kalmanSettingsInitialized || !aim_sim_kalman_settings_equal(s.kalmanSettingsSnapshot, settings);

    s.ctrlKalman.setSettings(settings);
    if (!changed)
        return;

    s.kalmanSettingsInitialized = true;
    s.kalmanSettingsSnapshot = settings;
    s.ctrlKalman.reset();
    s.controllerInitialized = false;
    s.ctrlPrevTimeSec = 0.0;
}

static void aim_sim_reset_wind_state(AimSimulationState& s)
{
    constexpr double twoPi = 6.28318530717958647692;
    std::uniform_real_distribution<double> phaseDist(0.0, twoPi);
    std::uniform_real_distribution<double> rateDist(0.04, 0.16);

    s.windCarryX = 0.0;
    s.windCarryY = 0.0;
    s.windVelX = 0.0;
    s.windVelY = 0.0;
    s.windNoiseX = 0.0;
    s.windNoiseY = 0.0;
    s.windFracX = 0.0;
    s.windFracY = 0.0;
    s.windPatternX = 0.0;
    s.windPatternY = 0.0;
    s.windPatternPhaseA = phaseDist(s.windRng);
    s.windPatternPhaseB = phaseDist(s.windRng);
    s.windPatternRateA = rateDist(s.windRng);
    s.windPatternRateB = rateDist(s.windRng);
}

static void aim_sim_sync_wind_config(AimSimulationState& s)
{
    const bool windEnabled = config.wind_mouse_enabled;
    const double windG = static_cast<double>(config.wind_G);
    const double windW = static_cast<double>(config.wind_W);
    const double windM = static_cast<double>(config.wind_M);
    const double windD = static_cast<double>(config.wind_D);

    const bool changed =
        (s.windEnabledSnapshot != windEnabled) ||
        (std::abs(s.windGSnapshot - windG) > 1e-6) ||
        (std::abs(s.windWSnapshot - windW) > 1e-6) ||
        (std::abs(s.windMSnapshot - windM) > 1e-6) ||
        (std::abs(s.windDSnapshot - windD) > 1e-6);

    if (!changed)
        return;

    aim_sim_reset_wind_state(s);
    s.windEnabledSnapshot = windEnabled;
    s.windGSnapshot = windG;
    s.windWSnapshot = windW;
    s.windMSnapshot = windM;
    s.windDSnapshot = windD;
}

static void aim_sim_choose_target_motion(AimSimulationState& s)
{
    const double stopChance = std::clamp(static_cast<double>(config.aim_sim_target_stop_chance), 0.0, 0.95);
    const double maxSpeed = std::max(20.0, static_cast<double>(config.aim_sim_target_max_speed));

    if (aim_sim_random_range(s, 0.0, 1.0) < stopChance)
    {
        s.targetDesiredVel = { 0.0, 0.0 };
        s.targetMotionRetargetSec = aim_sim_random_range(s, 0.25, 0.95);
        return;
    }

    const double angle = aim_sim_random_range(s, 0.0, 2.0 * 3.14159265358979323846);
    const double speed = aim_sim_random_range(s, maxSpeed * 0.25, maxSpeed);
    s.targetDesiredVel = { std::cos(angle) * speed, std::sin(angle) * speed };
    s.targetMotionRetargetSec = aim_sim_random_range(s, 0.35, 1.35);
}

static void aim_sim_reset(AimSimulationState& s, int panelW, int panelH)
{
    s.initialized = true;
    s.controllerInitialized = false;
    s.panelW = panelW;
    s.panelH = panelH;

    s.simTimeSec = 0.0;
    s.accumulatorSec = 0.0;
    s.currentFps = std::clamp((config.aim_sim_fps_min + config.aim_sim_fps_max) * 0.5, 15.0, 360.0);
    s.targetFps = s.currentFps;
    s.fpsRetargetSec = 0.0;
    s.targetMotionRetargetSec = 0.0;
    s.lastFrameDtSec = 1.0 / std::max(15.0, s.currentFps);

    s.targetPos = { panelW * 0.72, panelH * 0.38 };
    s.targetVel = { 0.0, 0.0 };
    s.targetDesiredVel = { 0.0, 0.0 };
    s.aimPos = { panelW * 0.50, panelH * 0.50 };
    s.observedTarget = s.targetPos;
    s.predictedTarget = s.targetPos;

    s.usedCaptureDelayMs = 0.0;
    s.usedInferenceDelayMs = 0.0;
    s.usedInputDelayMs = 0.0;
    s.usedExtraDelayMs = 0.0;
    s.usedTotalDelayMs = 0.0;
    s.lastErrorPx = 0.0;
    s.lastMoveX = 0;
    s.lastMoveY = 0;

    s.ctrlKalmanTelemetry = {};
    s.kalmanLookaheadSec = 0.0;
    s.kalmanInnovationPx = 0.0;
    s.kalmanMeasuredTarget = s.targetPos;
    s.kalmanEstimatedTarget = s.targetPos;
    s.kalmanVelocity = {};
    s.ctrlPrevTimeSec = 0.0;
    s.kalmanSettingsInitialized = false;
    s.ctrlKalman.reset();
    aim_sim_sync_kalman_config(s);

    aim_sim_reset_wind_state(s);
    s.windEnabledSnapshot = config.wind_mouse_enabled;
    s.windGSnapshot = static_cast<double>(config.wind_G);
    s.windWSnapshot = static_cast<double>(config.wind_W);
    s.windMSnapshot = static_cast<double>(config.wind_M);
    s.windDSnapshot = static_cast<double>(config.wind_D);

    s.history.clear();
    s.queuedMoves.clear();
    s.targetTrail.clear();
    s.aimTrail.clear();

    s.lastWallTime = std::chrono::steady_clock::now();
    aim_sim_choose_target_motion(s);
}

static double aim_sim_get_live_inference_ms()
{
    double delayMs = static_cast<double>(config.aim_sim_inference_delay_ms);

    if (config.backend == "DML")
    {
        if (dml_detector)
            delayMs = dml_detector->lastInferenceTimeDML.count();
    }
#ifdef USE_CUDA
    else
    {
        delayMs = trt_detector.lastInferenceTime.count();
    }
#endif

    if (!std::isfinite(delayMs))
        delayMs = static_cast<double>(config.aim_sim_inference_delay_ms);

    return std::clamp(delayMs, 0.0, 120.0);
}

static AimSimHistorySample aim_sim_sample_history(const std::deque<AimSimHistorySample>& history, double sampleTime)
{
    if (history.empty())
        return {};

    if (sampleTime <= history.front().timeSec)
        return history.front();
    if (sampleTime >= history.back().timeSec)
        return history.back();

    for (size_t i = 1; i < history.size(); ++i)
    {
        const auto& b = history[i];
        if (b.timeSec < sampleTime)
            continue;

        const auto& a = history[i - 1];
        const double dt = std::max(1e-6, b.timeSec - a.timeSec);
        const double t = std::clamp((sampleTime - a.timeSec) / dt, 0.0, 1.0);

        AimSimHistorySample out;
        out.timeSec = sampleTime;
        out.target.x = a.target.x + (b.target.x - a.target.x) * t;
        out.target.y = a.target.y + (b.target.y - a.target.y) * t;
        out.aim.x = a.aim.x + (b.aim.x - a.aim.x) * t;
        out.aim.y = a.aim.y + (b.aim.y - a.aim.y) * t;
        return out;
    }

    return history.back();
}

static double aim_sim_calculate_speed_multiplier(double distance, double maxDistance)
{
    const double snapRadius = std::max(0.0, static_cast<double>(config.snapRadius));
    const double nearRadius = std::max(1e-3, static_cast<double>(config.nearRadius));
    const double curveExp = std::max(0.1, static_cast<double>(config.speedCurveExponent));
    const double minSpeed = static_cast<double>(config.minSpeedMultiplier);
    const double maxSpeed = static_cast<double>(config.maxSpeedMultiplier);

    if (distance < snapRadius)
        return minSpeed * config.snapBoostFactor;

    if (distance < nearRadius)
    {
        const double t = distance / nearRadius;
        const double curve = 1.0 - std::pow(1.0 - t, curveExp);
        return minSpeed + (maxSpeed - minSpeed) * curve;
    }

    const double norm = std::clamp(distance / std::max(1e-6, maxDistance), 0.0, 1.0);
    return minSpeed + (maxSpeed - minSpeed) * norm;
}

static void aim_sim_enqueue_move(AimSimulationState& s, double executeAtSec, int mx, int my)
{
    if (mx == 0 && my == 0)
        return;

    // Match MouseThread queue behavior: small fixed queue with drop-oldest.
    constexpr size_t queueLimit = 5;
    if (s.queuedMoves.size() >= queueLimit)
        s.queuedMoves.pop_front();

    s.queuedMoves.push_back({ executeAtSec, mx, my });
}

static void aim_sim_enqueue_relative_path(AimSimulationState& s, double executeAtSec, int dx, int dy)
{
    if (dx == 0 && dy == 0)
        return;

    if (!config.wind_mouse_enabled)
    {
        aim_sim_enqueue_move(s, executeAtSec, dx, dy);
        return;
    }

    s.windCarryX += static_cast<double>(dx);
    s.windCarryY += static_cast<double>(dy);

    const double baseG = std::clamp(static_cast<double>(config.wind_G), 0.05, 50.0);
    const double baseW = std::clamp(static_cast<double>(config.wind_W), 0.0, 80.0);
    const double baseM = std::max(1.0, static_cast<double>(config.wind_M));
    const double baseD = std::max(1.0, static_cast<double>(config.wind_D));

    std::uniform_real_distribution<double> noiseDist(-1.0, 1.0);
    std::uniform_real_distribution<double> clipDist(0.55, 1.0);
    constexpr double twoPi = 6.28318530717958647692;

    const double carryMag = std::hypot(s.windCarryX, s.windCarryY);
    const int maxSubsteps = std::clamp(static_cast<int>(carryMag * 0.24) + 1, 1, 5);

    for (int i = 0; i < maxSubsteps; ++i)
    {
        const double dist = std::hypot(s.windCarryX, s.windCarryY);
        const double velMag = std::hypot(s.windVelX, s.windVelY);

        if (dist < 0.20 && velMag < 0.12)
            break;

        const double normDist = std::clamp(dist / baseD, 0.0, 1.0);
        const double pullGain = baseG * (0.25 + 0.75 * normDist);
        const double noiseAmp = baseW * (0.15 + 0.85 * normDist);

        double pullX = 0.0;
        double pullY = 0.0;
        if (dist > 1e-8)
        {
            pullX = s.windCarryX / dist * pullGain;
            pullY = s.windCarryY / dist * pullGain;
        }

        s.windPatternRateA = std::clamp(s.windPatternRateA + noiseDist(s.windRng) * 0.004, 0.025, 0.280);
        s.windPatternRateB = std::clamp(s.windPatternRateB + noiseDist(s.windRng) * 0.004, 0.025, 0.280);

        const double stepTempo = 0.20 + 0.95 * normDist;
        s.windPatternPhaseA += s.windPatternRateA * stepTempo;
        s.windPatternPhaseB += s.windPatternRateB * stepTempo;
        if (s.windPatternPhaseA > twoPi) s.windPatternPhaseA = std::fmod(s.windPatternPhaseA, twoPi);
        if (s.windPatternPhaseB > twoPi) s.windPatternPhaseB = std::fmod(s.windPatternPhaseB, twoPi);

        const double oscAX = std::sin(s.windPatternPhaseA);
        const double oscBX = std::sin(s.windPatternPhaseB + 1.61803398875);
        const double oscAY = std::cos(s.windPatternPhaseA * 0.79 + 0.35);
        const double oscBY = std::cos(s.windPatternPhaseB * 1.17 - 0.48);

        const double patternAmp = baseW * (0.05 + 0.55 * normDist);
        const double patternTargetX = (oscAX + 0.58 * oscBX) * patternAmp;
        const double patternTargetY = (oscAY + 0.58 * oscBY) * patternAmp;
        const double patternBlend = 0.12 + 0.20 * normDist;
        s.windPatternX = s.windPatternX * (1.0 - patternBlend) + patternTargetX * patternBlend;
        s.windPatternY = s.windPatternY * (1.0 - patternBlend) + patternTargetY * patternBlend;

        s.windNoiseX = s.windNoiseX * 0.72 + noiseDist(s.windRng) * noiseAmp * 0.28;
        s.windNoiseY = s.windNoiseY * 0.72 + noiseDist(s.windRng) * noiseAmp * 0.28;

        const double windForceX = s.windNoiseX + s.windPatternX * 0.42;
        const double windForceY = s.windNoiseY + s.windPatternY * 0.42;

        const double drag = 0.82 + (1.0 - normDist) * 0.10;
        s.windVelX = s.windVelX * drag + pullX + windForceX;
        s.windVelY = s.windVelY * drag + pullY + windForceY;

        const double vCap = std::max(0.65, baseM * (0.30 + 0.70 * normDist));
        const double newVelMag = std::hypot(s.windVelX, s.windVelY);
        if (newVelMag > vCap)
        {
            const double clip = vCap * clipDist(s.windRng);
            s.windVelX = (s.windVelX / newVelMag) * clip;
            s.windVelY = (s.windVelY / newVelMag) * clip;
        }

        s.windFracX += s.windVelX;
        s.windFracY += s.windVelY;

        const int stepX = static_cast<int>(std::round(s.windFracX));
        const int stepY = static_cast<int>(std::round(s.windFracY));
        if (stepX == 0 && stepY == 0)
            continue;

        s.windFracX -= static_cast<double>(stepX);
        s.windFracY -= static_cast<double>(stepY);
        s.windCarryX -= static_cast<double>(stepX);
        s.windCarryY -= static_cast<double>(stepY);
        aim_sim_enqueue_move(s, executeAtSec, stepX, stepY);
    }

    const double carryCap = 120.0;
    const double finalCarryMag = std::hypot(s.windCarryX, s.windCarryY);
    if (finalCarryMag > carryCap)
    {
        const double scale = carryCap / finalCarryMag;
        s.windCarryX *= scale;
        s.windCarryY *= scale;
    }
}

static AimSimVec2 aim_sim_counts_to_world_delta(int mx, int my, int detRes, double scaleToCtrlX, double scaleToCtrlY)
{
    if (detRes <= 0 || scaleToCtrlX <= 1e-8 || scaleToCtrlY <= 1e-8)
        return {};

    const Config::GameProfile* gpPtr = nullptr;
    auto activeIt = config.game_profiles.find(config.active_game);
    if (activeIt != config.game_profiles.end())
        gpPtr = &activeIt->second;
    else
    {
        auto unifiedIt = config.game_profiles.find("UNIFIED");
        if (unifiedIt != config.game_profiles.end())
            gpPtr = &unifiedIt->second;
    }
    if (!gpPtr)
        return {};
    const auto& gp = *gpPtr;

    if (gp.sens == 0.0 || gp.yaw == 0.0 || gp.pitch == 0.0)
        return {};

    const double fovNow = std::max(1.0, static_cast<double>(config.fovX));
    const double fovScale = (gp.fovScaled && gp.baseFOV > 1.0) ? (fovNow / gp.baseFOV) : 1.0;
    const double degX = static_cast<double>(mx) * gp.sens * gp.yaw * fovScale;
    const double degY = static_cast<double>(my) * gp.sens * gp.pitch * fovScale;

    const double degPerPxX = fovNow / static_cast<double>(detRes);
    const double degPerPxY = std::max(1e-6, static_cast<double>(config.fovY) / static_cast<double>(detRes));

    const double controlPxX = degX / degPerPxX;
    const double controlPxY = degY / degPerPxY;

    AimSimVec2 deltaWorld;
    deltaWorld.x = controlPxX / scaleToCtrlX;
    deltaWorld.y = controlPxY / scaleToCtrlY;
    return deltaWorld;
}

static double aim_sim_next_frame_dt(AimSimulationState& s)
{
    const double fpsMin = std::clamp(static_cast<double>(std::min(config.aim_sim_fps_min, config.aim_sim_fps_max)), 15.0, 360.0);
    const double fpsMax = std::clamp(static_cast<double>(std::max(config.aim_sim_fps_min, config.aim_sim_fps_max)), 15.0, 360.0);

    s.fpsRetargetSec -= s.lastFrameDtSec;
    if (s.fpsRetargetSec <= 0.0)
    {
        s.targetFps = aim_sim_random_range(s, fpsMin, fpsMax);
        s.fpsRetargetSec = aim_sim_random_range(s, 0.12, 0.55);
    }

    const double alpha = 1.0 - std::exp(-s.lastFrameDtSec * 5.0);
    s.currentFps += (s.targetFps - s.currentFps) * alpha;

    const double range = std::max(0.0, fpsMax - fpsMin);
    const double jitter = range * std::clamp(static_cast<double>(config.aim_sim_fps_jitter), 0.0, 0.8);
    const double instantFps = std::clamp(
        s.currentFps + aim_sim_random_range(s, -jitter, jitter),
        fpsMin, fpsMax
    );

    s.lastFrameDtSec = 1.0 / std::max(15.0, instantFps);
    return s.lastFrameDtSec;
}

static void aim_sim_step(AimSimulationState& s, double dtSec, int panelW, int panelH)
{
    s.simTimeSec += dtSec;
    aim_sim_sync_wind_config(s);
    aim_sim_sync_kalman_config(s);

    s.targetMotionRetargetSec -= dtSec;
    if (s.targetMotionRetargetSec <= 0.0)
        aim_sim_choose_target_motion(s);

    const double maxAccel = std::max(20.0, static_cast<double>(config.aim_sim_target_accel));
    const double dvX = s.targetDesiredVel.x - s.targetVel.x;
    const double dvY = s.targetDesiredVel.y - s.targetVel.y;
    const double dvLen = std::hypot(dvX, dvY);
    const double maxDv = maxAccel * dtSec;
    if (dvLen > maxDv && dvLen > 1e-8)
    {
        const double k = maxDv / dvLen;
        s.targetVel.x += dvX * k;
        s.targetVel.y += dvY * k;
    }
    else
    {
        s.targetVel.x = s.targetDesiredVel.x;
        s.targetVel.y = s.targetDesiredVel.y;
    }

    if (aim_sim_vec_len(s.targetDesiredVel) < 1.0)
    {
        const double damp = std::exp(-dtSec * 2.8);
        s.targetVel.x *= damp;
        s.targetVel.y *= damp;
    }

    s.targetPos.x += s.targetVel.x * dtSec;
    s.targetPos.y += s.targetVel.y * dtSec;

    const double margin = 10.0;
    const double maxX = std::max(margin, static_cast<double>(panelW) - margin);
    const double maxY = std::max(margin, static_cast<double>(panelH) - margin);

    if (s.targetPos.x < margin)
    {
        s.targetPos.x = margin;
        s.targetVel.x = std::abs(s.targetVel.x) * 0.65;
        s.targetDesiredVel.x = std::abs(s.targetDesiredVel.x);
    }
    else if (s.targetPos.x > maxX)
    {
        s.targetPos.x = maxX;
        s.targetVel.x = -std::abs(s.targetVel.x) * 0.65;
        s.targetDesiredVel.x = -std::abs(s.targetDesiredVel.x);
    }

    if (s.targetPos.y < margin)
    {
        s.targetPos.y = margin;
        s.targetVel.y = std::abs(s.targetVel.y) * 0.65;
        s.targetDesiredVel.y = std::abs(s.targetDesiredVel.y);
    }
    else if (s.targetPos.y > maxY)
    {
        s.targetPos.y = maxY;
        s.targetVel.y = -std::abs(s.targetVel.y) * 0.65;
        s.targetDesiredVel.y = -std::abs(s.targetDesiredVel.y);
    }

    s.history.push_back({ s.simTimeSec, s.targetPos, s.aimPos });
    while (!s.history.empty() && (s.simTimeSec - s.history.front().timeSec) > 3.0)
        s.history.pop_front();

    const double captureMs = std::clamp(static_cast<double>(config.aim_sim_capture_delay_ms), 0.0, 80.0);
    const double inferenceMs = config.aim_sim_use_live_inference
        ? aim_sim_get_live_inference_ms()
        : std::clamp(static_cast<double>(config.aim_sim_inference_delay_ms), 0.0, 120.0);
    const double inputMs = std::clamp(static_cast<double>(config.aim_sim_input_delay_ms), 0.0, 60.0);
    const double extraMs = std::clamp(static_cast<double>(config.aim_sim_extra_delay_ms), 0.0, 60.0);
    const double totalObsMs = captureMs + inferenceMs + extraMs;

    s.usedCaptureDelayMs = captureMs;
    s.usedInferenceDelayMs = inferenceMs;
    s.usedInputDelayMs = inputMs;
    s.usedExtraDelayMs = extraMs;
    s.usedTotalDelayMs = totalObsMs + inputMs;

    const double sampleTime = s.simTimeSec - (totalObsMs * 0.001);
    const AimSimHistorySample delayed = aim_sim_sample_history(s.history, sampleTime);
    s.observedTarget = delayed.target;

    const int detRes = std::max(32, config.detection_resolution);
    const double scaleToCtrlX = static_cast<double>(detRes) / std::max(1, panelW);
    const double scaleToCtrlY = static_cast<double>(detRes) / std::max(1, panelH);
    const double centerCtrlX = static_cast<double>(detRes) * 0.5;
    const double centerCtrlY = static_cast<double>(detRes) * 0.5;

    const double observedCtrlX = centerCtrlX + (delayed.target.x - delayed.aim.x) * scaleToCtrlX;
    const double observedCtrlY = centerCtrlY + (delayed.target.y - delayed.aim.y) * scaleToCtrlY;

    double predictedCtrlX = observedCtrlX;
    double predictedCtrlY = observedCtrlY;

    const double obsDt = s.controllerInitialized
        ? std::max(1e-4, s.simTimeSec - s.ctrlPrevTimeSec)
        : std::max(1e-4, dtSec);
    s.controllerInitialized = true;
    s.ctrlPrevTimeSec = s.simTimeSec;

    double lookaheadSec = std::max(0.0, static_cast<double>(config.predictionInterval));
    if (config.kalman_compensate_detection_delay)
        lookaheadSec += inferenceMs * 0.001;
    lookaheadSec += static_cast<double>(config.kalman_additional_prediction_ms) * 0.001;
    lookaheadSec = std::clamp(lookaheadSec, 0.0, 1.5);
    s.kalmanLookaheadSec = lookaheadSec;

    s.ctrlKalmanTelemetry = s.ctrlKalman.update(observedCtrlX, observedCtrlY, obsDt, lookaheadSec);
    predictedCtrlX = s.ctrlKalmanTelemetry.predicted_x;
    predictedCtrlY = s.ctrlKalmanTelemetry.predicted_y;

    if (!std::isfinite(predictedCtrlX))
        predictedCtrlX = observedCtrlX;
    if (!std::isfinite(predictedCtrlY))
        predictedCtrlY = observedCtrlY;

    s.kalmanMeasuredTarget.x = s.aimPos.x + (observedCtrlX - centerCtrlX) / scaleToCtrlX;
    s.kalmanMeasuredTarget.y = s.aimPos.y + (observedCtrlY - centerCtrlY) / scaleToCtrlY;
    s.kalmanEstimatedTarget.x = s.aimPos.x + (s.ctrlKalmanTelemetry.estimate_x - centerCtrlX) / scaleToCtrlX;
    s.kalmanEstimatedTarget.y = s.aimPos.y + (s.ctrlKalmanTelemetry.estimate_y - centerCtrlY) / scaleToCtrlY;
    s.kalmanMeasuredTarget.x = std::clamp(s.kalmanMeasuredTarget.x, 0.0, static_cast<double>(panelW));
    s.kalmanMeasuredTarget.y = std::clamp(s.kalmanMeasuredTarget.y, 0.0, static_cast<double>(panelH));
    s.kalmanEstimatedTarget.x = std::clamp(s.kalmanEstimatedTarget.x, 0.0, static_cast<double>(panelW));
    s.kalmanEstimatedTarget.y = std::clamp(s.kalmanEstimatedTarget.y, 0.0, static_cast<double>(panelH));
    s.kalmanVelocity.x = s.ctrlKalmanTelemetry.velocity_x / std::max(1e-8, scaleToCtrlX);
    s.kalmanVelocity.y = s.ctrlKalmanTelemetry.velocity_y / std::max(1e-8, scaleToCtrlY);
    const double innovationWorldX = s.ctrlKalmanTelemetry.innovation_x / std::max(1e-8, scaleToCtrlX);
    const double innovationWorldY = s.ctrlKalmanTelemetry.innovation_y / std::max(1e-8, scaleToCtrlY);
    s.kalmanInnovationPx = std::hypot(innovationWorldX, innovationWorldY);

    const double offsetX = predictedCtrlX - centerCtrlX;
    const double offsetY = predictedCtrlY - centerCtrlY;
    const double distancePx = std::hypot(offsetX, offsetY);

    s.lastMoveX = 0;
    s.lastMoveY = 0;

    if (distancePx > 0.0)
    {
        const double degPerPxX = static_cast<double>(config.fovX) / static_cast<double>(detRes);
        const double degPerPxY = static_cast<double>(config.fovY) / static_cast<double>(detRes);
        const double maxDistanceCtrl = std::hypot(static_cast<double>(detRes), static_cast<double>(detRes)) * 0.5;
        const double speed = aim_sim_calculate_speed_multiplier(distancePx, maxDistanceCtrl);
        const auto countsPair = config.degToCounts(offsetX * degPerPxX, offsetY * degPerPxY, config.fovX);

        double corr = 1.0;
        const double fps = 1.0 / std::max(1e-8, dtSec);
        if (fps > 30.0)
            corr = 30.0 / fps;

        const double moveX = countsPair.first * speed * corr;
        const double moveY = countsPair.second * speed * corr;

        const int mx = static_cast<int>(moveX);
        const int my = static_cast<int>(moveY);
        aim_sim_enqueue_relative_path(s, s.simTimeSec + inputMs * 0.001, mx, my);
    }

    while (!s.queuedMoves.empty() && s.queuedMoves.front().executeAtSec <= s.simTimeSec)
    {
        const AimSimQueuedMove m = s.queuedMoves.front();
        s.queuedMoves.pop_front();

        const AimSimVec2 worldDelta = aim_sim_counts_to_world_delta(m.mx, m.my, detRes, scaleToCtrlX, scaleToCtrlY);
        s.aimPos.x += worldDelta.x;
        s.aimPos.y += worldDelta.y;

        s.lastMoveX += m.mx;
        s.lastMoveY += m.my;
    }

    const double aimMargin = 6.0;
    s.aimPos.x = std::clamp(s.aimPos.x, aimMargin, std::max(aimMargin, static_cast<double>(panelW) - aimMargin));
    s.aimPos.y = std::clamp(s.aimPos.y, aimMargin, std::max(aimMargin, static_cast<double>(panelH) - aimMargin));

    s.predictedTarget.x = s.aimPos.x + (predictedCtrlX - centerCtrlX) / scaleToCtrlX;
    s.predictedTarget.y = s.aimPos.y + (predictedCtrlY - centerCtrlY) / scaleToCtrlY;
    s.predictedTarget.x = std::clamp(s.predictedTarget.x, 0.0, static_cast<double>(panelW));
    s.predictedTarget.y = std::clamp(s.predictedTarget.y, 0.0, static_cast<double>(panelH));

    s.lastErrorPx = std::hypot(s.targetPos.x - s.aimPos.x, s.targetPos.y - s.aimPos.y);

    if (config.aim_sim_show_history)
    {
        s.targetTrail.push_back(s.targetPos);
        s.aimTrail.push_back(s.aimPos);

        const size_t maxTrail = 160;
        while (s.targetTrail.size() > maxTrail) s.targetTrail.pop_front();
        while (s.aimTrail.size() > maxTrail) s.aimTrail.pop_front();
    }
    else
    {
        s.targetTrail.clear();
        s.aimTrail.clear();
    }
}

static void draw_aim_sim_panel(
    Game_overlay* overlay,
    AimSimulationState& s,
    int screenW,
    int screenH)
{
    if (!overlay || !config.aim_sim_enabled)
        return;

    const int panelW = std::clamp(config.aim_sim_width, 220, 1920);
    const int panelH = std::clamp(config.aim_sim_height, 180, 1080);
    int panelX = config.aim_sim_x;
    int panelY = config.aim_sim_y;

    const int minX = -panelW + 20;
    const int minY = -panelH + 20;
    const int maxX = std::max(20, screenW - 20);
    const int maxY = std::max(20, screenH - 20);
    panelX = std::clamp(panelX, minX, maxX);
    panelY = std::clamp(panelY, minY, maxY);

    if (!s.initialized || s.panelW != panelW || s.panelH != panelH)
        aim_sim_reset(s, panelW, panelH);

    auto now = std::chrono::steady_clock::now();
    double dtReal = 1.0 / 120.0;
    if (s.lastWallTime.time_since_epoch().count() != 0)
    {
        dtReal = std::chrono::duration<double>(now - s.lastWallTime).count();
        dtReal = std::clamp(dtReal, 0.001, 0.08);
    }
    s.lastWallTime = now;
    s.accumulatorSec = std::min(0.15, s.accumulatorSec + dtReal);

    int loops = 0;
    while (loops < 20)
    {
        const double frameDt = aim_sim_next_frame_dt(s);
        if (s.accumulatorSec < frameDt)
            break;
        aim_sim_step(s, frameDt, panelW, panelH);
        s.accumulatorSec -= frameDt;
        ++loops;
    }
    if (loops >= 20)
        s.accumulatorSec = 0.0;

    const float fx = static_cast<float>(panelX);
    const float fy = static_cast<float>(panelY);
    const float fw = static_cast<float>(panelW);
    const float fh = static_cast<float>(panelH);

    overlay->FillRect({ fx, fy, fw, fh }, ARGB(155, 12, 16, 20));
    overlay->AddRect({ fx, fy, fw, fh }, ARGB(230, 185, 190, 200), 1.5f);

    const float cx = fx + fw * 0.5f;
    const float cy = fy + fh * 0.5f;
    overlay->AddLine({ fx, cy, fx + fw, cy }, ARGB(90, 255, 255, 255), 1.0f);
    overlay->AddLine({ cx, fy, cx, fy + fh }, ARGB(90, 255, 255, 255), 1.0f);

    if (config.aim_sim_show_history && s.targetTrail.size() > 1)
    {
        const size_t n = s.targetTrail.size();
        for (size_t i = 1; i < n; ++i)
        {
            const float alpha = static_cast<float>(50 + (180 * i) / n);
            const auto& p0 = s.targetTrail[i - 1];
            const auto& p1 = s.targetTrail[i];
            overlay->AddLine(
                { fx + static_cast<float>(p0.x), fy + static_cast<float>(p0.y),
                  fx + static_cast<float>(p1.x), fy + static_cast<float>(p1.y) },
                ARGB(static_cast<uint8_t>(alpha), 255, 120, 120), 1.2f);
        }
    }

    if (config.aim_sim_show_history && s.aimTrail.size() > 1)
    {
        const size_t n = s.aimTrail.size();
        for (size_t i = 1; i < n; ++i)
        {
            const float alpha = static_cast<float>(50 + (180 * i) / n);
            const auto& p0 = s.aimTrail[i - 1];
            const auto& p1 = s.aimTrail[i];
            overlay->AddLine(
                { fx + static_cast<float>(p0.x), fy + static_cast<float>(p0.y),
                  fx + static_cast<float>(p1.x), fy + static_cast<float>(p1.y) },
                ARGB(static_cast<uint8_t>(alpha), 120, 220, 255), 1.2f);
        }
    }

    if (config.aim_sim_show_observed)
    {
        overlay->FillCircle(
            { fx + static_cast<float>(s.observedTarget.x), fy + static_cast<float>(s.observedTarget.y), 4.0f },
            ARGB(230, 255, 205, 90));
    }

    if (config.aim_sim_show_kalman_debug)
    {
        overlay->AddLine(
            { fx + static_cast<float>(s.kalmanMeasuredTarget.x), fy + static_cast<float>(s.kalmanMeasuredTarget.y),
              fx + static_cast<float>(s.kalmanEstimatedTarget.x), fy + static_cast<float>(s.kalmanEstimatedTarget.y) },
            ARGB(200, 110, 255, 205), 1.0f);
        overlay->FillCircle(
            { fx + static_cast<float>(s.kalmanEstimatedTarget.x), fy + static_cast<float>(s.kalmanEstimatedTarget.y), 3.5f },
            ARGB(220, 80, 255, 185));
        overlay->AddCircle(
            { fx + static_cast<float>(s.kalmanMeasuredTarget.x), fy + static_cast<float>(s.kalmanMeasuredTarget.y), 3.0f },
            ARGB(210, 255, 180, 80), 1.2f);

        const float velocityScale = 0.06f;
        const float velEndX = fx + static_cast<float>(s.kalmanEstimatedTarget.x + s.kalmanVelocity.x * velocityScale);
        const float velEndY = fy + static_cast<float>(s.kalmanEstimatedTarget.y + s.kalmanVelocity.y * velocityScale);
        overlay->AddLine(
            { fx + static_cast<float>(s.kalmanEstimatedTarget.x), fy + static_cast<float>(s.kalmanEstimatedTarget.y), velEndX, velEndY },
            ARGB(215, 120, 245, 255), 1.4f);
    }

    overlay->AddCircle(
        { fx + static_cast<float>(s.predictedTarget.x), fy + static_cast<float>(s.predictedTarget.y), 6.0f },
        ARGB(210, 255, 245, 100), 1.5f);

    overlay->FillCircle(
        { fx + static_cast<float>(s.targetPos.x), fy + static_cast<float>(s.targetPos.y), 5.0f },
        ARGB(250, 255, 90, 90));

    overlay->FillCircle(
        { fx + static_cast<float>(s.aimPos.x), fy + static_cast<float>(s.aimPos.y), 4.0f },
        ARGB(255, 80, 200, 255));
    overlay->AddCircle(
        { fx + static_cast<float>(s.aimPos.x), fy + static_cast<float>(s.aimPos.y), 9.0f },
        ARGB(235, 80, 200, 255), 1.8f);

    overlay->AddLine(
        { fx + static_cast<float>(s.aimPos.x), fy + static_cast<float>(s.aimPos.y),
          fx + static_cast<float>(s.targetPos.x), fy + static_cast<float>(s.targetPos.y) },
        ARGB(160, 255, 255, 255), 1.0f);

    const float tx = fx + 10.0f;
    float ty = fy + 8.0f;
    const float step = 17.0f;

    overlay->AddText(tx, ty, L"Aim Simulation", 16.0f, ARGB(245, 230, 235, 245));
    ty += step;

    wchar_t line[256]{};
    const double simFps = 1.0 / std::max(1e-6, s.lastFrameDtSec);
    swprintf_s(line, L"FPS %.1f (range %d..%d)", simFps, config.aim_sim_fps_min, config.aim_sim_fps_max);
    overlay->AddText(tx, ty, line, 14.0f, ARGB(220, 210, 220, 230));
    ty += step;

    swprintf_s(line, L"Delay %.1f ms (cap %.1f + inf %.1f + in %.1f + extra %.1f)",
        s.usedTotalDelayMs, s.usedCaptureDelayMs, s.usedInferenceDelayMs, s.usedInputDelayMs, s.usedExtraDelayMs);
    overlay->AddText(tx, ty, line, 14.0f, ARGB(220, 210, 220, 230));
    ty += step;

    swprintf_s(line, L"Target speed %.0f px/s | Error %.1f px",
        aim_sim_vec_len(s.targetVel), s.lastErrorPx);
    overlay->AddText(tx, ty, line, 14.0f, ARGB(220, 210, 220, 230));
    ty += step;

    swprintf_s(line, L"Move counts dx=%d dy=%d", s.lastMoveX, s.lastMoveY);
    overlay->AddText(tx, ty, line, 14.0f, ARGB(220, 210, 220, 230));

    if (config.aim_sim_show_kalman_debug)
    {
        ty += step;
        swprintf_s(line, L"Kalman %.3fs | innovation %.2f px", s.kalmanLookaheadSec, s.kalmanInnovationPx);
        overlay->AddText(tx, ty, line, 14.0f, ARGB(215, 190, 240, 230));
    }
}



void gameOverlayRenderLoop()
{
    const int vx = GetSystemMetrics(SM_XVIRTUALSCREEN);
    const int vy = GetSystemMetrics(SM_YVIRTUALSCREEN);

    MONITORINFO mi{};
    mi.cbSize = sizeof(mi);
    HMONITOR hPrimary = MonitorFromPoint(POINT{ 0,0 }, MONITOR_DEFAULTTOPRIMARY);
    GetMonitorInfo(hPrimary, &mi);
    RECT pr = mi.rcMonitor;
    const int pw = pr.right - pr.left;
    const int ph = pr.bottom - pr.top;

    const int offX = pr.left - vx;
    const int offY = pr.top - vy;

#ifdef USE_CUDA
    static depth_anything::DepthAnythingTrt depthDebugModel;
    static std::string depthDebugModelPath;
    static int depthDebugColormap = -1;
    static int depthDebugImageId = 0;
    static int depthMaskImageId = 0;
    static cv::Mat depthDebugFrame;
    static auto lastDepthUpdate = std::chrono::steady_clock::time_point::min();
    static bool lastDepthInferenceEnabled = true;
#endif
    int lastDetectionVersion = -1;
    static AimSimulationState aimSimState;

    while (!gameOverlayShouldExit.load())
    {
        if (!config.game_overlay_enabled)
        {
            aimSimState.initialized = false;
            if (gameOverlayPtr)
            {
                gameOverlayPtr->Stop();
                delete gameOverlayPtr;
                gameOverlayPtr = nullptr;
                std::lock_guard<std::mutex> lk(g_iconMutex);
                g_lastIconPath.clear();
                g_iconImageId = 0;
                g_iconLastError.clear();
            }
#ifdef USE_CUDA
            depthDebugModel.reset();
            depthDebugModelPath.clear();
            depthDebugColormap = -1;
            depthDebugImageId = 0;
            depthMaskImageId = 0;
            depthDebugFrame.release();
            lastDepthUpdate = std::chrono::steady_clock::time_point::min();
#endif
            std::this_thread::sleep_for(std::chrono::milliseconds(150));
            continue;
        }

        if (!gameOverlayPtr)
        {
            gameOverlayPtr = new Game_overlay();
            gameOverlayPtr->SetWindowBounds(0, 0, pw, ph);
            gameOverlayPtr->SetMaxFPS(config.game_overlay_max_fps > 0 ? (unsigned)config.game_overlay_max_fps : 0);
            gameOverlayPtr->SetExcludeFromCapture(config.overlay_exclude_from_capture);
            gameOverlayPtr->Start();
        }
        else if (!gameOverlayPtr->IsRunning())
        {
            gameOverlayPtr->SetWindowBounds(0, 0, pw, ph);
            gameOverlayPtr->SetMaxFPS(config.game_overlay_max_fps > 0 ? (unsigned)config.game_overlay_max_fps : 0);
            gameOverlayPtr->SetExcludeFromCapture(config.overlay_exclude_from_capture);
            gameOverlayPtr->Start();
        }

        if (!gameOverlayPtr || !gameOverlayPtr->IsRunning())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(150));
            continue;
        }

        gameOverlayPtr->SetMaxFPS(config.game_overlay_max_fps > 0 ? (unsigned)config.game_overlay_max_fps : 0);
        gameOverlayPtr->SetExcludeFromCapture(config.overlay_exclude_from_capture);

        const int detRes = config.detection_resolution;
        if (detRes <= 0)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        int regionW = detRes;
        int regionH = detRes;

        if (regionW > pw) regionW = pw;
        if (regionH > ph) regionH = ph;

        const int baseX = (pw - regionW) / 2;
        const int baseY = (ph - regionH) / 2;

        const float scaleX = detRes > 0 ? (static_cast<float>(regionW) / static_cast<float>(detRes)) : 1.0f;
        const float scaleY = detRes > 0 ? (static_cast<float>(regionH) / static_cast<float>(detRes)) : 1.0f;

        std::vector<cv::Rect> boxesCopy;
        std::vector<int> classesCopy;
        int detectionVersion = lastDetectionVersion;
        {
            std::unique_lock<std::mutex> lk(detectionBuffer.mutex);
            const unsigned fpsCap = (unsigned)config.game_overlay_max_fps;
            const int waitMs = (fpsCap > 0) ? static_cast<int>(std::max(1u, 1000u / fpsCap)) : 8;
            detectionBuffer.cv.wait_for(lk, std::chrono::milliseconds(waitMs), [&] {
                return detectionBuffer.version != lastDetectionVersion || gameOverlayShouldExit.load();
                });
            boxesCopy = detectionBuffer.boxes;
            classesCopy = detectionBuffer.classes;
            detectionVersion = detectionBuffer.version;
        }
        lastDetectionVersion = detectionVersion;

        decltype(globalMouseThread->getFuturePositions()) futurePts;
        if (config.game_overlay_draw_future && globalMouseThread)
            futurePts = globalMouseThread->getFuturePositions();

        std::vector<std::pair<double, double>> windTailPts;
        if (config.game_overlay_draw_wind_tail && globalMouseThread)
            windTailPts = globalMouseThread->getWindDebugTrail();

        std::vector<TrackDebugInfo> trackDebugCopy;
        int lockedTrackId = -1;
        {
            std::lock_guard<std::mutex> lk(g_trackerDebugMutex);
            trackDebugCopy = g_trackerDebugTracks;
            lockedTrackId = g_trackerLockedId;
        }

        if (config.game_overlay_icon_enabled)
        {
            std::lock_guard<std::mutex> lk(g_iconMutex);
            if (config.game_overlay_icon_path != g_lastIconPath)
            {
                if (g_iconImageId != 0)
                {
                    gameOverlayPtr->UnloadImage(g_iconImageId);
                    g_iconImageId = 0;
                }
                g_lastIconPath = config.game_overlay_icon_path;
                std::error_code fsErr;
                std::filesystem::path p;
                try
                {
                    p = std::filesystem::u8path(g_lastIconPath);
                }
                catch (const std::exception&)
                {
                    p = std::filesystem::path(g_lastIconPath);
                }
                const bool hasFile = !g_lastIconPath.empty() && p.has_filename() && std::filesystem::is_regular_file(p, fsErr);
                if (fsErr)
                {
                    g_iconImageId = 0;
                    g_iconLastError = "[GameOverlay] Failed to read icon path: " + g_lastIconPath + " (" + fsErr.message() + ")";
                    std::cerr << g_iconLastError << std::endl;
                }
                else if (hasFile)
                {
                    const std::wstring wpath = p.wstring();
                    g_iconLastError.clear();

                    UINT iw = 0, ih = 0;
                    std::string verr;
                    if (!IsValidImageFile(wpath, iw, ih, verr))
                    {
                        g_iconImageId = 0;
                        g_iconLastError = "[GameOverlay] Invalid image '" + g_lastIconPath + "': " + verr;
                        std::cerr << g_iconLastError << std::endl;
                    }
                    else
                    {
                        try
                        {
                            int id = gameOverlayPtr->LoadImageFromFile(wpath);
                            if (id != 0)
                            {
                                g_iconImageId = id;
                                std::cout << "[GameOverlay] Loaded icon (" << iw << "x" << ih << "): " << g_lastIconPath << std::endl;
                            }
                            else
                            {
                                g_iconImageId = 0;
                                g_iconLastError = "[GameOverlay] Failed to load icon (loader returned 0): " + g_lastIconPath;
                                std::cerr << g_iconLastError << std::endl;
                            }
                        }
                        catch (const std::exception& e)
                        {
                            g_iconImageId = 0;
                            g_iconLastError = std::string("[GameOverlay] Exception while loading icon: ") + e.what();
                            std::cerr << g_iconLastError << std::endl;
                        }
                        catch (...)
                        {
                            g_iconImageId = 0;
                            g_iconLastError = "[GameOverlay] Unknown exception while loading icon.";
                            std::cerr << g_iconLastError << std::endl;
                        }
                    }
                }
                else
                {
                    g_iconImageId = 0;
                    g_iconLastError = "[GameOverlay] Icon file not found: " + g_lastIconPath;
                    std::cerr << g_iconLastError << std::endl;
                }
            }
        }

        gameOverlayPtr->BeginFrame();

#ifdef USE_CUDA
        if (!config.depth_inference_enabled)
        {
            if (lastDepthInferenceEnabled)
            {
                if (gameOverlayPtr)
                {
                    if (depthDebugImageId != 0)
                    {
                        gameOverlayPtr->UnloadImage(depthDebugImageId);
                        depthDebugImageId = 0;
                    }
                    if (depthMaskImageId != 0)
                    {
                        gameOverlayPtr->UnloadImage(depthMaskImageId);
                        depthMaskImageId = 0;
                    }
                }

                depthDebugModel.reset();
                depthDebugModelPath.clear();
                depthDebugColormap = -1;
                depthDebugFrame.release();
                lastDepthUpdate = std::chrono::steady_clock::time_point::min();

                auto& depthMask = depth_anything::GetDepthMaskGenerator();
                depthMask.reset();
            }
            lastDepthInferenceEnabled = false;
        }
        else
        {
            lastDepthInferenceEnabled = true;

            float depthW = 0.0f;
            float depthH = 0.0f;
            float maskW = 0.0f;
            float maskH = 0.0f;
            float maskOpacity = std::clamp(static_cast<float>(config.depth_mask_alpha) / 255.0f, 0.0f, 1.0f);
            bool maskHasBounds = false;
            cv::Rect maskBounds{};

            if (config.depth_debug_overlay_enabled)
            {
                cv::Mat frameCopy;
                {
                    std::lock_guard<std::mutex> lk(frameMutex);
                    if (!latestFrame.empty())
                        latestFrame.copyTo(frameCopy);
                }

                if (config.depth_model_path.empty())
                {
                    if (depthDebugModel.ready())
                        depthDebugModel.reset();
                    depthDebugModelPath.clear();
                }
                else if (depthDebugModelPath != config.depth_model_path || !depthDebugModel.ready())
                {
                    if (depthDebugModel.initialize(config.depth_model_path, gLogger))
                    {
                        depthDebugModelPath = config.depth_model_path;
                    }
                }

                if (config.depth_colormap != depthDebugColormap)
                {
                    depthDebugModel.setColormap(config.depth_colormap);
                    depthDebugColormap = config.depth_colormap;
                }

                if (depthDebugModel.ready() && !frameCopy.empty())
                {
                    auto now = std::chrono::steady_clock::now();
                    bool shouldUpdate = depthDebugFrame.empty();
                    if (config.depth_fps <= 0)
                    {
                        shouldUpdate = true;
                    }
                    else if (!shouldUpdate)
                    {
                        auto interval = std::chrono::milliseconds(1000 / config.depth_fps);
                        shouldUpdate = (now - lastDepthUpdate) >= interval;
                    }
                    if (shouldUpdate)
                    {
                        cv::Mat depthFrame = depthDebugModel.predict(frameCopy);
                        if (!depthFrame.empty())
                        {
                            depthDebugFrame = depthFrame;
                            lastDepthUpdate = now;
                        }
                    }
                }
                if (!depthDebugFrame.empty())
                {
                    cv::Mat depthBGRA;
                    cv::cvtColor(depthDebugFrame, depthBGRA, cv::COLOR_BGR2BGRA);
                    int newId = gameOverlayPtr->UpdateImageFromBGRA(
                        depthBGRA.data,
                        depthBGRA.cols,
                        depthBGRA.rows,
                        static_cast<int>(depthBGRA.step),
                        depthDebugImageId);
                    if (newId != 0)
                        depthDebugImageId = newId;
                    depthW = static_cast<float>(regionW);
                    depthH = static_cast<float>(regionH);
                }
            }
            else if (depthDebugImageId != 0)
            {
                gameOverlayPtr->UnloadImage(depthDebugImageId);
                depthDebugImageId = 0;
            }

            if (config.depth_mask_enabled)
            {
                auto& depthMask = depth_anything::GetDepthMaskGenerator();
                cv::Mat mask = depthMask.getMask();

                if (mask.empty())
                {
                    cv::Mat frameCopy;
                    {
                        std::lock_guard<std::mutex> lk(frameMutex);
                        if (!latestFrame.empty())
                            latestFrame.copyTo(frameCopy);
                    }

                    if (!frameCopy.empty())
                    {
                        depth_anything::DepthMaskOptions maskOptions;
                        maskOptions.enabled = true;
                        maskOptions.fps = config.depth_mask_fps;
                        maskOptions.near_percent = config.depth_mask_near_percent;
                        maskOptions.expand = config.depth_mask_expand;
                        maskOptions.invert = config.depth_mask_invert;

                        depthMask.update(frameCopy, maskOptions, config.depth_model_path, gLogger);
                        mask = depthMask.getMask();

                        if (mask.empty())
                        {
                            if (!config.depth_model_path.empty() &&
                                (depthDebugModelPath != config.depth_model_path || !depthDebugModel.ready()))
                            {
                                if (depthDebugModel.initialize(config.depth_model_path, gLogger))
                                {
                                    depthDebugModelPath = config.depth_model_path;
                                    depthDebugColormap = config.depth_colormap;
                                    depthDebugModel.setColormap(config.depth_colormap);
                                }
                            }

                            if (depthDebugModel.ready())
                            {
                                cv::Mat depthLocal = depthDebugModel.predictDepth(frameCopy);
                                if (!depthLocal.empty())
                                {
                                    const int nearPercent = std::clamp(config.depth_mask_near_percent, 1, 100);
                                    const bool invertMask = config.depth_mask_invert;
                                    const int total = depthLocal.rows * depthLocal.cols;
                                    if (total > 0)
                                    {
                                        int hist[256] = {};
                                        for (int y = 0; y < depthLocal.rows; ++y)
                                        {
                                            const uint8_t* row = depthLocal.ptr<uint8_t>(y);
                                            for (int x = 0; x < depthLocal.cols; ++x)
                                                hist[row[x]]++;
                                        }

                                        const int target = std::max(1, (total * nearPercent) / 100);
                                        int threshold = 0;
                                        if (!invertMask)
                                        {
                                            int count = 0;
                                            for (int i = 0; i < 256; ++i)
                                            {
                                                count += hist[i];
                                                if (count >= target)
                                                {
                                                    threshold = i;
                                                    break;
                                                }
                                            }
                                            cv::compare(depthLocal, threshold, mask, cv::CMP_LE);
                                        }
                                        else
                                        {
                                            int count = 0;
                                            for (int i = 255; i >= 0; --i)
                                            {
                                                count += hist[i];
                                                if (count >= target)
                                                {
                                                    threshold = i;
                                                    break;
                                                }
                                            }
                                            cv::compare(depthLocal, threshold, mask, cv::CMP_GE);
                                        }

                                        const int expand = std::clamp(config.depth_mask_expand, 0, 128);
                                        if (expand > 0)
                                        {
                                            const int kernelSize = 2 * expand + 1;
                                            cv::Mat kernel = cv::getStructuringElement(
                                                cv::MORPH_ELLIPSE, cv::Size(kernelSize, kernelSize));
                                            cv::dilate(mask, mask, kernel);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                if (!mask.empty())
                {
                    cv::Mat maskBGRA(mask.size(), CV_8UC4, cv::Scalar(0, 0, 0, 0));
                    maskBGRA.setTo(cv::Scalar(20, 90, 255, 255), mask);

                    cv::Mat nonZeroPoints;
                    cv::findNonZero(mask, nonZeroPoints);
                    if (!nonZeroPoints.empty())
                    {
                        maskBounds = cv::boundingRect(nonZeroPoints);
                        maskHasBounds = true;

                    }

                    int newId = gameOverlayPtr->UpdateImageFromBGRA(
                        maskBGRA.data,
                        maskBGRA.cols,
                        maskBGRA.rows,
                        static_cast<int>(maskBGRA.step),
                        depthMaskImageId);
                    if (newId != 0)
                        depthMaskImageId = newId;

                    maskW = static_cast<float>(regionW);
                    maskH = static_cast<float>(regionH);
                }
            }
            else if (depthMaskImageId != 0)
            {
                gameOverlayPtr->UnloadImage(depthMaskImageId);
                depthMaskImageId = 0;
            }

            if (depthDebugImageId != 0 || depthMaskImageId != 0 || (config.depth_debug_overlay_enabled && config.depth_mask_enabled))
            {
                float depthX = static_cast<float>(baseX);
                float depthY = static_cast<float>(baseY);
                float maskX = depthX;
                float maskY = depthY;

                if (depthDebugImageId != 0 && depthW > 0.0f && depthH > 0.0f)
                {
                    const float depthDebugOpacity = config.depth_mask_enabled ? 0.30f : 1.0f;
                    gameOverlayPtr->DrawImage(depthDebugImageId, depthX, depthY, depthW, depthH, depthDebugOpacity);
                    gameOverlayPtr->AddRect({ depthX, depthY, depthW, depthH }, ARGB(120, 255, 255, 255), 1.0f);
                }

                if (depthMaskImageId != 0 && maskW > 0.0f && maskH > 0.0f)
                {
                    gameOverlayPtr->DrawImage(depthMaskImageId, maskX, maskY, maskW, maskH, maskOpacity);

                    if (maskHasBounds)
                    {
                        const float bx = maskX + static_cast<float>(maskBounds.x) * scaleX;
                        const float by = maskY + static_cast<float>(maskBounds.y) * scaleY;
                        const float bw = static_cast<float>(maskBounds.width) * scaleX;
                        const float bh = static_cast<float>(maskBounds.height) * scaleY;

                        gameOverlayPtr->AddRect({ bx, by, bw, bh }, ARGB(230, 255, 240, 120), 1.8f);
                    }
                }
            }
        }
#endif

        // CAPTURE FRAME
        if (config.game_overlay_draw_frame)
        {
            int A = config.game_overlay_frame_a;
            int R = config.game_overlay_frame_r;
            int G = config.game_overlay_frame_g;
            int B = config.game_overlay_frame_b;
            auto clamp255 = [](int& v) { if (v < 0) v = 0; else if (v > 255) v = 255; };
            clamp255(A); clamp255(R); clamp255(G); clamp255(B);
            const uint32_t col =
                (uint32_t(A) << 24) |
                (uint32_t(R) << 16) |
                (uint32_t(G) << 8) |
                uint32_t(B);

            float thickness = config.game_overlay_frame_thickness;
            if (thickness <= 0.f) thickness = 1.f;

            if (config.circle_mask)
            {
                float cx = baseX + regionW * 0.5f;
                float cy = baseY + regionH * 0.5f;
                float radius = std::min(regionW, regionH) * 0.5f;
                gameOverlayPtr->AddCircle({ cx, cy, radius }, col, thickness);
            }
            else
            {
                gameOverlayPtr->AddRect(
                    { static_cast<float>(baseX),
                      static_cast<float>(baseY),
                      static_cast<float>(regionW),
                      static_cast<float>(regionH) },
                    col, thickness);
            }
        }

        // DYNAMIC FOV VISUALIZER
        if (config.game_overlay_draw_frame) 
        {
            float cx = baseX + regionW * 0.5f;
            float cy = baseY + regionH * 0.5f;
            float currentFovW = static_cast<float>(config.fovX) * scaleX;
            float currentFovH = static_cast<float>(config.fovY) * scaleY;
            
            // Draws a red box showing your exact active FOV
            gameOverlayPtr->AddRect(
                { cx - (currentFovW * 0.5f), cy - (currentFovH * 0.5f), currentFovW, currentFovH }, 
                ARGB(255, 255, 50, 50), // Red color
                1.5f // Thickness
            );
        }

        // BOXES
        if (config.game_overlay_draw_boxes && (!boxesCopy.empty() || !trackDebugCopy.empty()))
        {
            int A = config.game_overlay_box_a;
            int R = config.game_overlay_box_r;
            int G = config.game_overlay_box_g;
            int B = config.game_overlay_box_b;
            auto clamp255 = [](int& v) { if (v < 0) v = 0; else if (v > 255) v = 255; };
            clamp255(A); clamp255(R); clamp255(G); clamp255(B);
            const uint32_t col =
                (uint32_t(A) << 24) |
                (uint32_t(R) << 16) |
                (uint32_t(G) << 8) |
                uint32_t(B);

            float thickness = config.game_overlay_box_thickness;
            if (thickness <= 0.f) thickness = 1.f;

            for (const auto& b : boxesCopy)
            {
                if (b.width <= 0 || b.height <= 0) continue;

                int bx = std::max(0, std::min(b.x, detRes));
                int by = std::max(0, std::min(b.y, detRes));
                int bw = std::max(0, std::min(b.width, detRes - bx));
                int bh = std::max(0, std::min(b.height, detRes - by));
                if (bw == 0 || bh == 0) continue;

                float x = baseX + bx * scaleX;
                float y = baseY + by * scaleY;
                float w = bw * scaleX;
                float h = bh * scaleY;

                if (x + w < baseX || y + h < baseY ||
                    x > baseX + regionW || y > baseY + regionH)
                    continue;

                gameOverlayPtr->AddRect({ x, y, w, h }, col, thickness);
            }

            for (const auto& t : trackDebugCopy)
            {
                const auto& b = t.box;
                if (b.width <= 0 || b.height <= 0) continue;

                int bx = std::max(0, std::min(b.x, detRes));
                int by = std::max(0, std::min(b.y, detRes));
                int bw = std::max(0, std::min(b.width, detRes - bx));
                int bh = std::max(0, std::min(b.height, detRes - by));
                if (bw == 0 || bh == 0) continue;

                const float x = baseX + bx * scaleX;
                const float y = baseY + by * scaleY;

                std::wstring label = L"ID " + std::to_wstring(t.trackId);
                if (t.trackId == lockedTrackId || t.isLocked)
                    label += L" *";
                if (!t.observedThisFrame)
                    label += L" m" + std::to_wstring(t.missedFrames);

                const uint32_t textCol =
                    (t.trackId == lockedTrackId || t.isLocked)
                    ? ARGB(255, 255, 220, 70)
                    : ARGB(230, 180, 255, 180);

                gameOverlayPtr->AddText(
                    x + 2.0f,
                    std::max(static_cast<float>(baseY), y - 16.0f),
                    label,
                    15.0f,
                    textCol
                );
            }
        }

        // FUTURE POINTS
        if (config.game_overlay_draw_future && !futurePts.empty())
        {
            const int total = static_cast<int>(futurePts.size());
            const int baseA = std::max(5, std::min(255, config.game_overlay_box_a));

            for (int i = 0; i < total; ++i)
            {
                float alphaFactor =
                    std::exp(-config.game_overlay_future_alpha_falloff *
                        (static_cast<float>(i) / static_cast<float>(total)));

                int a = static_cast<int>(baseA * alphaFactor);
                if (a < 12) a = 12;

                const uint32_t col =
                    (uint32_t(a) << 24) |
                    (uint32_t(255 - (i * 255 / total)) << 16) |
                    (uint32_t(50) << 8) |
                    (uint32_t(i * 255 / total));

                float px = static_cast<float>(baseX) + static_cast<float>(futurePts[i].first) * scaleX;
                float py = static_cast<float>(baseY) + static_cast<float>(futurePts[i].second) * scaleY;

                if (px < baseX - 40 || py < baseY - 40 ||
                    px > baseX + regionW + 40 || py > baseY + regionH + 40)
                    continue;

                gameOverlayPtr->FillCircle({ px, py, config.game_overlay_future_point_radius }, col);
            }
        }

        // WIND DEBUG TAIL
        if (config.game_overlay_draw_wind_tail && windTailPts.size() > 1)
        {
            const size_t n = windTailPts.size();
            const auto& anchor = windTailPts.back();
            const float centerX = static_cast<float>(baseX) + regionW * 0.5f;
            const float centerY = static_cast<float>(baseY) + regionH * 0.5f;
            for (size_t i = 1; i < n; ++i)
            {
                const auto& p0 = windTailPts[i - 1];
                const auto& p1 = windTailPts[i];

                const float rel0x = static_cast<float>(p0.first - anchor.first);
                const float rel0y = static_cast<float>(p0.second - anchor.second);
                const float rel1x = static_cast<float>(p1.first - anchor.first);
                const float rel1y = static_cast<float>(p1.second - anchor.second);

                const float x0 = centerX + rel0x * scaleX;
                const float y0 = centerY + rel0y * scaleY;
                const float x1 = centerX + rel1x * scaleX;
                const float y1 = centerY + rel1y * scaleY;

                const uint8_t alpha = static_cast<uint8_t>(35 + (190 * i) / n);
                gameOverlayPtr->AddLine({ x0, y0, x1, y1 }, ARGB(alpha, 80, 210, 255), 1.3f);
            }

            const float hx = centerX;
            const float hy = centerY;
            gameOverlayPtr->FillCircle({ hx, hy, 3.5f }, ARGB(230, 120, 230, 255));
            gameOverlayPtr->AddText(
                static_cast<float>(baseX) + 8.0f,
                static_cast<float>(baseY + regionH) - 22.0f,
                L"Wind tail",
                14.0f,
                ARGB(210, 120, 230, 255)
            );
        }

        // ICONS
        if (config.game_overlay_icon_enabled && g_iconImageId != 0 && !boxesCopy.empty())
        {
            const int iconW = config.game_overlay_icon_width;
            const int iconH = config.game_overlay_icon_height;
            const float offXIcon = config.game_overlay_icon_offset_x;
            const float offYIcon = config.game_overlay_icon_offset_y;
            std::string anchor = config.game_overlay_icon_anchor;
            const int wantedClass = config.game_overlay_icon_class;
            const size_t count = boxesCopy.size();
            for (size_t i = 0; i < count; ++i)
            {
                const auto& b = boxesCopy[i];
                int cls = (i < classesCopy.size()) ? classesCopy[i] : -1;
                // Class filter (-1 = all)
                if (wantedClass >= 0 && cls != wantedClass)
                {
                    continue;
                }

                if (b.width <= 0 || b.height <= 0) continue;
                int bx = std::max(0, std::min(b.x, detRes));
                int by = std::max(0, std::min(b.y, detRes));
                int bw = std::max(0, std::min(b.width, detRes - bx));
                int bh = std::max(0, std::min(b.height, detRes - by));
                if (bw == 0 || bh == 0) continue;

                float boxX = baseX + bx * scaleX;
                float boxY = baseY + by * scaleY;
                float boxW = bw * scaleX;
                float boxH = bh * scaleY;

                float drawX = boxX;
                float drawY = boxY;

                if (anchor == "center")
                {
                    drawX = boxX + boxW / 2.0f - iconW / 2.0f;
                    drawY = boxY + boxH / 2.0f - iconH / 2.0f;
                }
                else if (anchor == "top" || anchor == "head")
                {
                    drawX = boxX + boxW / 2.0f - iconW / 2.0f;
                    drawY = boxY - iconH;
                }
                else if (anchor == "bottom")
                {
                    drawX = boxX + boxW / 2.0f - iconW / 2.0f;
                    drawY = boxY + boxH;
                }
                else
                {
                    drawX = boxX + boxW / 2.0f - iconW / 2.0f;
                    drawY = boxY + boxH / 2.0f - iconH / 2.0f;
                }

                drawX += offXIcon;
                drawY += offYIcon;

                gameOverlayPtr->DrawImage(g_iconImageId, drawX, drawY, (float)iconW, (float)iconH, 1.0f);
            }
        }

        if (config.game_overlay_show_target_correction)
        {
            draw_target_correction_demo_game_overlay(
                gameOverlayPtr,
                static_cast<float>(baseX) + regionW * 0.5f,
                static_cast<float>(baseY) + regionH * 0.5f);
        }

        if (config.aim_sim_enabled)
        {
            draw_aim_sim_panel(gameOverlayPtr, aimSimState, pw, ph);
        }
        else
        {
            aimSimState.initialized = false;
        }

        gameOverlayPtr->EndFrame();
    }

    if (gameOverlayPtr)
    {
        gameOverlayPtr->Stop();
        delete gameOverlayPtr;
        gameOverlayPtr = nullptr;
    }
}
