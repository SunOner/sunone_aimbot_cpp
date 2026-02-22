#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <iostream>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <exception>
#include <filesystem>
#include <algorithm>
#include <cmath>
#include <optional>
#include <deque>
#include <random>
#include <array>
#include <cwchar>

#include "capture.h"
#include "mouse.h"
#include "sunone_aimbot_cpp.h"
#include "keyboard_listener.h"
#include "overlay.h"
#include "Game_overlay.h"
#include "ghub.h"
#include "other_tools.h"
#include "virtual_camera.h"
#include "mem/gpu_resource_manager.h"
#include "mem/cpu_affinity_manager.h"

#ifdef USE_CUDA
#include "depth/depth_anything_trt.h"
#include "depth/depth_mask.h"
#include "tensorrt/nvinf.h"
#endif

std::condition_variable frameCV;
std::atomic<bool> shouldExit(false);
std::atomic<bool> aiming(false);
std::atomic<bool> detectionPaused(false);
std::mutex configMutex;

#ifdef USE_CUDA
TrtDetector trt_detector;
#endif

DirectMLDetector* dml_detector = nullptr;
MouseThread* globalMouseThread = nullptr;
Config config;

Game_overlay* gameOverlayPtr = nullptr;
std::thread gameOverlayThread;
std::atomic<bool> gameOverlayShouldExit(false);

GhubMouse* gHub = nullptr;
Arduino* arduinoSerial = nullptr;
KmboxNetConnection* kmboxNetSerial = nullptr;
KmboxAConnection* kmboxASerial = nullptr;
MakcuConnection* makcuSerial = nullptr;

std::atomic<bool> detection_resolution_changed(false);
std::atomic<bool> capture_method_changed(false);
std::atomic<bool> capture_cursor_changed(false);
std::atomic<bool> capture_borders_changed(false);
std::atomic<bool> capture_fps_changed(false);
std::atomic<bool> capture_window_changed(false);
std::atomic<bool> detector_model_changed(false);
std::atomic<bool> show_window_changed(false);
std::atomic<bool> input_method_changed(false);

std::atomic<bool> zooming(false);
std::atomic<bool> shooting(false);

static std::string g_lastIconPath;
static int g_iconImageId = 0;
static std::mutex g_iconMutex;
static std::mutex g_trackerDebugMutex;
static std::vector<TrackDebugInfo> g_trackerDebugTracks;
static int g_trackerLockedId = -1;

std::string g_iconLastError;

static int FatalExit(const std::string& message)
{
    std::cerr << message << std::endl;
    std::cout << "Press Enter to exit...";
    std::cin.get();
    return -1;
}

static void HandleThreadCrash(const char* name, const std::exception* ex)
{
    std::cerr << "[Thread] " << name << " crashed: "
              << (ex ? ex->what() : "unknown exception") << std::endl;
    shouldExit = true;
    gameOverlayShouldExit.store(true);
    detectionBuffer.cv.notify_all();
}

template <typename Func>
static std::thread StartThreadGuarded(const char* name, Func func)
{
    return std::thread([name, func]() mutable {
        try
        {
            func();
        }
        catch (const std::exception& e)
        {
            HandleThreadCrash(name, &e);
        }
        catch (...)
        {
            HandleThreadCrash(name, nullptr);
        }
        });
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

    // Controller state matched to current MouseThread::moveMousePivot.
    double ctrlPrevX = 0.0;
    double ctrlPrevY = 0.0;
    double ctrlPrevVelocityX = 0.0;
    double ctrlPrevVelocityY = 0.0;
    double ctrlPrevTimeSec = 0.0;

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

    s.ctrlPrevX = 0.0;
    s.ctrlPrevY = 0.0;
    s.ctrlPrevVelocityX = 0.0;
    s.ctrlPrevVelocityY = 0.0;
    s.ctrlPrevTimeSec = 0.0;

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

    // Wind micro-steps generated in one sim tick share executeAtSec.
    // Coalesce them to avoid queue starvation when input delay is enabled.
    if (!s.queuedMoves.empty())
    {
        AimSimQueuedMove& back = s.queuedMoves.back();
        if (std::abs(back.executeAtSec - executeAtSec) <= 1e-6)
        {
            back.mx += mx;
            back.my += my;
            return;
        }
    }

    // Simulation applies queued moves after input delay, so queue must cover
    // all moves produced during that delay window at max configured FPS.
    const double inputDelayMs = std::clamp(static_cast<double>(config.aim_sim_input_delay_ms), 0.0, 60.0);
    const double maxFps = std::clamp(
        static_cast<double>(std::max(config.aim_sim_fps_min, config.aim_sim_fps_max)),
        15.0, 360.0);
    const double minFrameMs = 1000.0 / std::max(1.0, maxFps);
    const double delayedFrames = inputDelayMs / std::max(0.1, minFrameMs);
    const double stepsPerFrame = config.wind_mouse_enabled ? 5.0 : 1.0;
    const int needed = static_cast<int>(std::ceil((delayedFrames + 2.0) * stepsPerFrame));
    const size_t queueLimit = static_cast<size_t>(std::clamp(needed, 32, 256));

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

    if (!s.controllerInitialized)
    {
        s.controllerInitialized = true;
        s.ctrlPrevX = observedCtrlX;
        s.ctrlPrevY = observedCtrlY;
        s.ctrlPrevVelocityX = 0.0;
        s.ctrlPrevVelocityY = 0.0;
        s.ctrlPrevTimeSec = s.simTimeSec;
    }
    else
    {
        const double obsDt = std::max(s.simTimeSec - s.ctrlPrevTimeSec, 1e-8);
        const double vx = std::clamp((observedCtrlX - s.ctrlPrevX) / obsDt, -20000.0, 20000.0);
        const double vy = std::clamp((observedCtrlY - s.ctrlPrevY) / obsDt, -20000.0, 20000.0);

        s.ctrlPrevX = observedCtrlX;
        s.ctrlPrevY = observedCtrlY;
        s.ctrlPrevVelocityX = vx;
        s.ctrlPrevVelocityY = vy;
        s.ctrlPrevTimeSec = s.simTimeSec;

        predictedCtrlX = observedCtrlX + vx * static_cast<double>(config.predictionInterval) + vx * 0.002;
        predictedCtrlY = observedCtrlY + vy * static_cast<double>(config.predictionInterval) + vy * 0.002;
    }

    if (!std::isfinite(predictedCtrlX))
        predictedCtrlX = observedCtrlX;
    if (!std::isfinite(predictedCtrlY))
        predictedCtrlY = observedCtrlY;

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
}


void createInputDevices()
{
    if (arduinoSerial)
    {
        delete arduinoSerial;
        arduinoSerial = nullptr;
    }

    if (gHub)
    {
        gHub->mouse_close();
        delete gHub;
        gHub = nullptr;
    }

    if (kmboxNetSerial)
    {
        delete kmboxNetSerial;
        kmboxNetSerial = nullptr;
    }

    if (kmboxASerial)
    {
        delete kmboxASerial;
        kmboxASerial = nullptr;
    }

    if (makcuSerial)
    {
        delete makcuSerial;
        makcuSerial = nullptr;
    }

    if (config.input_method == "ARDUINO")
    {
        std::cout << "[Mouse] Using Arduino method input." << std::endl;
        arduinoSerial = new Arduino(config.arduino_port, config.arduino_baudrate);
    }
    else if (config.input_method == "GHUB")
    {
        std::cout << "[Mouse] Using Ghub method input." << std::endl;
        gHub = new GhubMouse();
        if (!gHub->mouse_xy(0, 0))
        {
            std::cerr << "[Ghub] Error with opening mouse." << std::endl;
            delete gHub;
            gHub = nullptr;
        }
    }
    else if (config.input_method == "KMBOX_NET")
    {
        std::cout << "[Mouse] Using KMBOX_NET input." << std::endl;
        kmboxNetSerial = new KmboxNetConnection(config.kmbox_net_ip, config.kmbox_net_port, config.kmbox_net_uuid);
        if (!kmboxNetSerial->isOpen())
        {
            std::cerr << "[KmboxNet] Error connecting." << std::endl;
            delete kmboxNetSerial; kmboxNetSerial = nullptr;
        }
    }
    else if (config.input_method == "KMBOX_A")
    {
        std::cout << "[Mouse] Using KMBOX_A input." << std::endl;
        if (config.kmbox_a_pidvid.empty())
        {
            std::cerr << "[KmboxA] PIDVID is empty." << std::endl;
            return;
        }
        kmboxASerial = new KmboxAConnection(config.kmbox_a_pidvid);
        if (!kmboxASerial->isOpen())
        {
            std::cerr << "[KmboxA] Error connecting." << std::endl;
            delete kmboxASerial;
            kmboxASerial = nullptr;
        }
    }
    else if (config.input_method == "MAKCU")
    {
        std::cout << "[Mouse] Using MAKCU input." << std::endl;
        makcuSerial = new MakcuConnection(config.makcu_port, config.makcu_baudrate);
        if (!makcuSerial->isOpen())
        {
            std::cerr << "[Makcu] Error connecting." << std::endl;
            delete makcuSerial;
            makcuSerial = nullptr;
        }
    }
    else
    {
        std::cout << "[Mouse] Using default Win32 method input." << std::endl;
    }
}

void assignInputDevices()
{
    if (globalMouseThread)
    {
        globalMouseThread->setArduinoConnection(arduinoSerial);
        globalMouseThread->setGHubMouse(gHub);
        globalMouseThread->setKmboxAConnection(kmboxASerial);
        globalMouseThread->setKmboxNetConnection(kmboxNetSerial);
        globalMouseThread->setMakcuConnection(makcuSerial);
    }
}

void handleEasyNoRecoil(MouseThread& mouseThread)
{
    if (config.easynorecoil && shooting.load() && zooming.load())
    {
        std::lock_guard<std::mutex> lock(mouseThread.input_method_mutex);
        int recoil_compensation = static_cast<int>(config.easynorecoilstrength);

        if (arduinoSerial)
        {
            arduinoSerial->move(0, recoil_compensation);
        }
        else if (gHub)
        {
            gHub->mouse_xy(0, recoil_compensation);
        }
        else if (kmboxNetSerial)
        {
            kmboxNetSerial->move(0, recoil_compensation);
        }
        else if (kmboxASerial)
        {
            kmboxASerial->move(0, recoil_compensation);
        }
        else if (makcuSerial)
        {
            makcuSerial->move(0, recoil_compensation);
        }
        else
        {
            INPUT input = { 0 };
            input.type = INPUT_MOUSE;
            input.mi.dx = 0;
            input.mi.dy = recoil_compensation;
            input.mi.dwFlags = MOUSEEVENTF_MOVE | MOUSEEVENTF_VIRTUALDESK;
            SendInput(1, &input, sizeof(INPUT));
        }
    }
}

void mouseThreadFunction(MouseThread& mouseThread)
{
    int lastVersion = -1;
    std::vector<cv::Rect> boxes;
    std::vector<int> classes;
    MultiTargetTracker targetTracker;
    std::optional<AimbotTarget> activeTarget;
    auto lastTrackerUpdate = std::chrono::steady_clock::time_point::min();

    while (!shouldExit)
    {
        bool hasNewDetection = false;
        bool hasAimObservation = false;

        {
            std::unique_lock<std::mutex> lock(detectionBuffer.mutex);
            detectionBuffer.cv.wait_for(lock, std::chrono::milliseconds(1), [&] {
                return detectionBuffer.version > lastVersion || shouldExit;
                }
            );

            if (shouldExit) break;

            if (detectionBuffer.version > lastVersion)
            {
                boxes = detectionBuffer.boxes;
                classes = detectionBuffer.classes;
                lastVersion = detectionBuffer.version;
                hasNewDetection = true;
            }
        }

        if (input_method_changed.load())
        {
            createInputDevices();
            assignInputDevices();
            input_method_changed.store(false);
        }

        if (detection_resolution_changed.load())
        {
            {
                std::lock_guard<std::mutex> cfgLock(configMutex);
                mouseThread.updateConfig(
                    config.detection_resolution,
                    config.fovX,
                    config.fovY,
                    config.minSpeedMultiplier,
                    config.maxSpeedMultiplier,
                    config.predictionInterval,
                    config.auto_shoot,
                    config.bScope_multiplier
                );
            }
            targetTracker.reset();
            {
                std::lock_guard<std::mutex> lk(g_trackerDebugMutex);
                g_trackerDebugTracks.clear();
                g_trackerLockedId = -1;
            }
            detection_resolution_changed.store(false);
        }

        if (hasNewDetection)
        {
            targetTracker.update(
                boxes,
                classes,
                config.detection_resolution,
                config.detection_resolution,
                config.disable_headshot,
                aiming.load()
            );
            lastTrackerUpdate = std::chrono::steady_clock::now();
            {
                std::lock_guard<std::mutex> lk(g_trackerDebugMutex);
                g_trackerDebugTracks = targetTracker.getDebugTracks();
                g_trackerLockedId = targetTracker.getLockedTrackId();
            }

            LockedTargetInfo lockInfo;
            if (targetTracker.getLockedTarget(lockInfo) && lockInfo.observedThisFrame)
            {
                activeTarget = lockInfo.target;
                hasAimObservation = true;
                mouseThread.setLastTargetTime(std::chrono::steady_clock::now());
                mouseThread.setTargetDetected(true);

                auto futurePositions = mouseThread.predictFuturePositions(
                    activeTarget->pivotX,
                    activeTarget->pivotY,
                    config.prediction_futurePositions
                );
                mouseThread.storeFuturePositions(futurePositions);
            }
            else
            {
                activeTarget.reset();
                mouseThread.clearFuturePositions();
                mouseThread.setTargetDetected(false);
                mouseThread.clearQueuedMoves();
            }
        }

        if (activeTarget)
        {
            const int fps = std::max(1, captureFps.load());
            const int staleMs = std::clamp(2000 / fps, 25, 180);
            if (std::chrono::steady_clock::now() - lastTrackerUpdate > std::chrono::milliseconds(staleMs))
            {
                activeTarget.reset();
                mouseThread.clearFuturePositions();
                mouseThread.setTargetDetected(false);
                mouseThread.clearQueuedMoves();
            }
        }

        if (aiming)
        {
            if (activeTarget && hasAimObservation)
            {
                mouseThread.moveMousePivot(activeTarget->pivotX, activeTarget->pivotY);

                if (config.auto_shoot)
                {
                    mouseThread.pressMouse(*activeTarget);
                }
            }
            else
            {
                if (!activeTarget)
                {
                    mouseThread.clearQueuedMoves();
                }

                if (config.auto_shoot)
                {
                    mouseThread.releaseMouse();
                }
            }
        }
        else
        {
            mouseThread.clearQueuedMoves();
            if (config.auto_shoot)
            {
                mouseThread.releaseMouse();
            }
        }

        handleEasyNoRecoil(mouseThread);

        mouseThread.checkAndResetPredictions();
    }
}

static void gameOverlayRenderLoop()
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

                float depthW = 0.0f;
                float depthH = 0.0f;
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

                float maskW = 0.0f;
                float maskH = 0.0f;
                if (config.depth_mask_enabled)
                {
                    auto& depthMask = depth_anything::GetDepthMaskGenerator();
                    cv::Mat mask = depthMask.getMask();
                    if (!mask.empty())
                    {
                        cv::Mat alpha(mask.size(), CV_8U, cv::Scalar(0));
                        alpha.setTo(config.depth_mask_alpha, mask);

                        std::vector<cv::Mat> channels(4);
                        channels[0] = alpha;
                        channels[1] = cv::Mat(mask.size(), CV_8U, cv::Scalar(0));
                        channels[2] = cv::Mat(mask.size(), CV_8U, cv::Scalar(0));
                        channels[3] = alpha;

                        cv::Mat maskBGRA;
                        cv::merge(channels, maskBGRA);

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

                if (depthDebugImageId != 0 || depthMaskImageId != 0)
                {
                    const float pad = 8.0f;
                    float depthX = static_cast<float>(baseX);
                    float depthY = static_cast<float>(baseY);
                    float maskX = depthX;
                    float maskY = depthY;

                    if (depthDebugImageId != 0 && depthW > 0.0f && depthH > 0.0f)
                    {
                        gameOverlayPtr->DrawImage(depthDebugImageId, depthX, depthY, depthW, depthH, 1.0f);
                        gameOverlayPtr->AddRect({ depthX, depthY, depthW, depthH }, ARGB(120, 255, 255, 255), 1.0f);
                    }

                    if (depthMaskImageId != 0 && maskW > 0.0f && maskH > 0.0f)
                    {
                        gameOverlayPtr->DrawImage(depthMaskImageId, maskX, maskY, maskW, maskH, 1.0f);
                        gameOverlayPtr->AddText(maskX + pad, maskY + pad + 18.0f, L"Depth mask", 16.0f, ARGB(220, 255, 255, 255));
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

int main()
{
    SetConsoleOutputCP(CP_UTF8);
    SetRandomConsoleTitle();
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);

    {
        wchar_t exePath[MAX_PATH]{};
        if (GetModuleFileNameW(nullptr, exePath, MAX_PATH) > 0)
        {
            std::filesystem::path exeDir = std::filesystem::path(exePath).parent_path();
            std::error_code ec;
            std::filesystem::current_path(exeDir, ec);
            if (ec && config.verbose)
            {
                std::cout << "[Config] Failed to set working dir: " << exeDir.u8string()
                          << " (" << ec.message() << ")" << std::endl;
            }
        }
    }

    if (!config.loadConfig())
    {
        std::cerr << "[Config] Error with loading config!" << std::endl;
        std::cin.get();
        return -1;
    }

    CPUAffinityManager cpuManager;

    if (config.cpuCoreReserveCount > 0)
    {
        if (!cpuManager.reserveCPUCores(config.cpuCoreReserveCount))
            return FatalExit("[MAIN] Failed to reserve CPU cores.");
    }

    if (config.systemMemoryReserveMB > 0)
    {
        if (!cpuManager.reserveSystemMemory(config.systemMemoryReserveMB))
            return FatalExit("[MAIN] Failed to reserve system memory.");
    }

    try
    {
#ifdef USE_CUDA
        int cuda_runtime_version = 0;
        cudaError_t runtime_status = cudaRuntimeGetVersion(&cuda_runtime_version);

        if (runtime_status != cudaSuccess)
        {
            std::cerr << "[MAIN] CUDA runtime check failed: " << cudaGetErrorString(runtime_status) << std::endl;
            std::cin.get();
            return -1;
        }

        if (config.verbose)
            std::cout << "[CUDA] Version: " << cuda_runtime_version << std::endl;

        const int required_cuda_version = 13010;
        if (cuda_runtime_version < required_cuda_version)
        {
            int required_major = required_cuda_version / 1000;
            int required_minor = (required_cuda_version % 1000) / 10;
            int runtime_major = cuda_runtime_version / 1000;
            int runtime_minor = (cuda_runtime_version % 1000) / 10;
            std::cerr << "[MAIN] CUDA " << required_major << "." << required_minor
                << " required. Detected " << runtime_major << "." << runtime_minor << "." << std::endl;
            const wchar_t* title = L"CUDA Update Required";
            std::wstring message =
                L"An outdated CUDA version was detected. "
                L"Please update your graphics drivers to the latest version "
                L"and install CUDA 13.1.\n\n"
                L"The program will now attempt to continue.";
            MessageBoxW(nullptr, message.c_str(), title, MB_OK | MB_ICONWARNING);
        }

        GPUResourceManager gpuManager;
        if (config.gpuMemoryReserveMB > 0)
        {
            if (!gpuManager.reserveGPUMemory(config.gpuMemoryReserveMB))
                return FatalExit("[MAIN] Failed to reserve GPU memory.");
        }
        
        if (config.enableGpuExclusiveMode)
        {
            if (!gpuManager.setGPUExclusiveMode())
                return FatalExit("[MAIN] Failed to set GPU exclusive mode.");
        }

        int cuda_devices = 0;
        if (cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0)
        {
            std::cerr << "[MAIN] CUDA required but no devices found." << std::endl;
            std::cin.get();
            return -1;
        }
#endif
        if (!CreateDirectory(L"screenshots", NULL) && GetLastError() != ERROR_ALREADY_EXISTS)
        {
            std::cout << "[MAIN] Error with screenshot folder" << std::endl;
            std::cin.get();
            return -1;
        }

        if (!CreateDirectory(L"models", NULL) && GetLastError() != ERROR_ALREADY_EXISTS)
        {
            std::cout << "[MAIN] Error with models folder" << std::endl;
            std::cin.get();
            return -1;
        }
        if (!CreateDirectory(L"models\\depth", NULL) && GetLastError() != ERROR_ALREADY_EXISTS)
        {
            std::cout << "[MAIN] Error with models\\depth folder" << std::endl;
            std::cin.get();
            return -1;
        }

        if (config.capture_method == "virtual_camera")
        {
            auto cams = VirtualCameraCapture::GetAvailableVirtualCameras();
            if (!cams.empty())
            {
                if (config.virtual_camera_name == "None" ||
                    std::find(cams.begin(), cams.end(), config.virtual_camera_name) == cams.end())
                {
                    config.virtual_camera_name = cams[0];
                    config.saveConfig("config.ini");
                    std::cout << "[MAIN] Set virtual_camera_name = " << config.virtual_camera_name << std::endl;
                }
                std::cout << "[MAIN] Virtual cameras loaded: " << cams.size() << std::endl;
            }
            else
            {
                std::cerr << "[MAIN] No virtual cameras found" << std::endl;
            }
        }

        std::string modelPath = "models/" + config.ai_model;

        if (!std::filesystem::exists(modelPath))
        {
            std::cerr << "[MAIN] Specified model does not exist: " << modelPath << std::endl;

            std::vector<std::string> modelFiles = getModelFiles();

            if (!modelFiles.empty())
            {
                config.ai_model = modelFiles[0];
                config.saveConfig();
                std::cout << "[MAIN] Loaded first available model: " << config.ai_model << std::endl;
            }
            else
            {
                std::cerr << "[MAIN] No models found in 'models' directory." << std::endl;
                std::cin.get();
                return -1;
            }
        }

        createInputDevices();

        MouseThread mouseThread(
            config.detection_resolution,
            config.fovX,
            config.fovY,
            config.minSpeedMultiplier,
            config.maxSpeedMultiplier,
            config.predictionInterval,
            config.auto_shoot,
            config.bScope_multiplier,
            arduinoSerial,
            gHub,
            kmboxASerial,
            kmboxNetSerial,
            makcuSerial
        );

        globalMouseThread = &mouseThread;
        assignInputDevices();

        std::vector<std::string> availableModels = getAvailableModels();

        if (!config.ai_model.empty())
        {
            std::string modelPath = "models/" + config.ai_model;
            if (!std::filesystem::exists(modelPath))
            {
                std::cerr << "[MAIN] Specified model does not exist: " << modelPath << std::endl;

                if (!availableModels.empty())
                {
                    config.ai_model = availableModels[0];
                    config.saveConfig("config.ini");
                    std::cout << "[MAIN] Loaded first available model: " << config.ai_model << std::endl;
                }
                else
                {
                    std::cerr << "[MAIN] No models found in 'models' directory." << std::endl;
                    std::cin.get();
                    return -1;
                }
            }
        }
        else
        {
            if (!availableModels.empty())
            {
                config.ai_model = availableModels[0];
                config.saveConfig();
                std::cout << "[MAIN] No AI model specified in config. Loaded first available model: " << config.ai_model << std::endl;
            }
            else
            {
                std::cerr << "[MAIN] No AI models found in 'models' directory." << std::endl;
                std::cin.get();
                return -1;
            }
        }

        std::thread dml_detThread;

        if (config.backend == "DML")
        {
            dml_detector = new DirectMLDetector("models/" + config.ai_model);
            std::cout << "[MAIN] DML detector initialized." << std::endl;
            dml_detThread = StartThreadGuarded("DmlDetector", [] {
                dml_detector->dmlInferenceThread();
                });
        }
#ifdef USE_CUDA
        else
        {
            trt_detector.initialize("models/" + config.ai_model);
        }
#endif

        detection_resolution_changed.store(true);

        std::thread keyThread = StartThreadGuarded("KeyboardListener", [] {
            keyboardListener();
            });
        std::thread capThread = StartThreadGuarded("CaptureThread", [] {
            captureThread(config.detection_resolution, config.detection_resolution);
            });

#ifdef USE_CUDA
        std::thread trt_detThread = StartThreadGuarded("TrtDetector", [] {
            trt_detector.inferenceThread();
            });
#endif
        std::thread mouseMovThread = StartThreadGuarded("MouseThread", [&mouseThread] {
            mouseThreadFunction(mouseThread);
            });
        std::thread overlayThread = StartThreadGuarded("OverlayThread", [] {
            OverlayThread();
            });

        gameOverlayShouldExit.store(false);
        gameOverlayThread = StartThreadGuarded("GameOverlay", [] {
            gameOverlayRenderLoop();
            });

        welcome_message();

        keyThread.join();
        capThread.join();
        if (dml_detThread.joinable())
        {
            dml_detector->shouldExit = true;
            dml_detector->inferenceCV.notify_all();
            dml_detThread.join();
        }

#ifdef USE_CUDA
        trt_detThread.join();
#endif
        mouseMovThread.join();
        overlayThread.join();

        if (arduinoSerial)
        {
            delete arduinoSerial;
        }

        if (gHub)
        {
            gHub->mouse_close();
            delete gHub;
        }

        if (kmboxASerial)
        {
            delete kmboxASerial;
            kmboxASerial = nullptr;
        }

        if (dml_detector)
        {
            delete dml_detector;
            dml_detector = nullptr;
        }

        gameOverlayShouldExit.store(true);
        if (gameOverlayThread.joinable()) gameOverlayThread.join();
        if (gameOverlayPtr)
        {
            gameOverlayPtr->Stop();
            delete gameOverlayPtr;
            gameOverlayPtr = nullptr;
        }

        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[MAIN] An error has occurred in the main stream: " << e.what() << std::endl;
        std::cout << "Press Enter to exit...";
        std::cin.get();
        return -1;
    }
}
