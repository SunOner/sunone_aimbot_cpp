#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <cmath>
#include <algorithm>
#include <chrono>
#include <mutex>
#include <atomic>
#include <vector>
#include <iostream>
#include <random>

#include "mouse.h"
#include "capture.h"
#include "Arduino.h"
#include "sunone_aimbot_2.h"
#include "ghub.h"

MouseThread::MouseThread(
    int resolution,
    int fovX,
    int fovY,
    double minSpeedMultiplier,
    double maxSpeedMultiplier,
    double predictionInterval,
    bool auto_shoot,
    float bScope_multiplier,
    Arduino* arduinoConnection,
    GhubMouse* gHubMouse,
    KmboxAConnection* Kmbox_A_Connection,
    KmboxNetConnection* Kmbox_Net_Connection,
    MakcuConnection* makcuConnection)
    : screen_width(resolution),
    screen_height(resolution),
    prediction_interval(predictionInterval),
    fov_x(fovX),
    fov_y(fovY),
    max_distance(std::hypot(resolution, resolution) / 2.0),
    min_speed_multiplier(minSpeedMultiplier),
    max_speed_multiplier(maxSpeedMultiplier),
    center_x(resolution / 2.0),
    center_y(resolution / 2.0),
    auto_shoot(auto_shoot),
    bScope_multiplier(bScope_multiplier),
    arduino(arduinoConnection),
    kmbox_a(Kmbox_A_Connection),
    kmbox_net(Kmbox_Net_Connection),
    makcu(makcuConnection),
    gHub(gHubMouse),

    prev_velocity_x(0.0),
    prev_velocity_y(0.0),
    prev_x(0.0),
    prev_y(0.0)
{
    prev_time = std::chrono::steady_clock::time_point();
    last_target_time = std::chrono::steady_clock::now();

    wind_mouse_enabled = config.wind_mouse_enabled;
    wind_G = config.wind_G;
    wind_W = config.wind_W;
    wind_M = config.wind_M;
    wind_D = config.wind_D;
    resetWindState();
    clearWindDebugTrail();

    moveWorker = std::thread(&MouseThread::moveWorkerLoop, this);
}

void MouseThread::updateConfig(
    int resolution,
    int fovX,
    int fovY,
    double minSpeedMultiplier,
    double maxSpeedMultiplier,
    double predictionInterval,
    bool auto_shoot,
    float bScope_multiplier
)
{
    screen_width = screen_height = resolution;
    fov_x = fovX;  fov_y = fovY;
    min_speed_multiplier = minSpeedMultiplier;
    max_speed_multiplier = maxSpeedMultiplier;
    prediction_interval = predictionInterval;
    this->auto_shoot = auto_shoot;
    this->bScope_multiplier = bScope_multiplier;

    center_x = center_y = resolution / 2.0;
    max_distance = std::hypot(resolution, resolution) / 2.0;

    wind_mouse_enabled = config.wind_mouse_enabled;
    wind_G = config.wind_G; wind_W = config.wind_W;
    wind_M = config.wind_M; wind_D = config.wind_D;
    resetWindState();
    clearWindDebugTrail();
}

MouseThread::~MouseThread()
{
    workerStop = true;
    queueCv.notify_all();
    if (moveWorker.joinable()) moveWorker.join();
}

void MouseThread::queueMove(int dx, int dy)
{
    if (dx == 0 && dy == 0)
    {
        return;
    }

    std::lock_guard lg(queueMtx);
    if (moveQueue.size() >= queueLimit) moveQueue.pop();
    moveQueue.push({ dx,dy });
    queueCv.notify_one();
}

void MouseThread::moveWorkerLoop()
{
    try
    {
        while (!workerStop)
        {
            std::unique_lock ul(queueMtx);
            queueCv.wait(ul, [&] { return workerStop || !moveQueue.empty(); });

            while (!moveQueue.empty())
            {
                Move m = moveQueue.front();
                moveQueue.pop();
                ul.unlock();
                sendMovementToDriver(m.dx, m.dy);
                appendWindDebugStep(m.dx, m.dy);
                ul.lock();
            }
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "[Mouse] Move worker crashed: " << e.what() << std::endl;
    }
    catch (...)
    {
        std::cerr << "[Mouse] Move worker crashed: unknown exception." << std::endl;
    }
}

void MouseThread::windMouseMoveRelative(int dx, int dy)
{
    if (dx == 0 && dy == 0)
        return;

    windCarryX += static_cast<double>(dx);
    windCarryY += static_cast<double>(dy);

    const double baseG = std::clamp(wind_G, 0.05, 50.0);
    const double baseW = std::clamp(wind_W, 0.0, 80.0);
    const double baseM = std::max(1.0, wind_M);
    const double baseD = std::max(1.0, wind_D);

    std::uniform_real_distribution<double> noiseDist(-1.0, 1.0);
    std::uniform_real_distribution<double> clipDist(0.55, 1.0);
    constexpr double twoPi = 6.28318530717958647692;

    const double carryMag = std::hypot(windCarryX, windCarryY);
    const int maxSubsteps = std::clamp(static_cast<int>(carryMag * 0.24) + 1, 1, 5);

    for (int i = 0; i < maxSubsteps; ++i)
    {
        const double dist = std::hypot(windCarryX, windCarryY);
        const double velMag = std::hypot(windVelX, windVelY);

        if (dist < 0.20 && velMag < 0.12)
            break;

        const double normDist = std::clamp(dist / baseD, 0.0, 1.0);
        const double pullGain = baseG * (0.25 + 0.75 * normDist);
        const double noiseAmp = baseW * (0.15 + 0.85 * normDist);

        double pullX = 0.0;
        double pullY = 0.0;
        if (dist > 1e-8)
        {
            pullX = windCarryX / dist * pullGain;
            pullY = windCarryY / dist * pullGain;
        }

        windPatternRateA = std::clamp(windPatternRateA + noiseDist(windRng) * 0.004, 0.025, 0.280);
        windPatternRateB = std::clamp(windPatternRateB + noiseDist(windRng) * 0.004, 0.025, 0.280);

        const double stepTempo = 0.20 + 0.95 * normDist;
        windPatternPhaseA += windPatternRateA * stepTempo;
        windPatternPhaseB += windPatternRateB * stepTempo;
        if (windPatternPhaseA > twoPi) windPatternPhaseA = std::fmod(windPatternPhaseA, twoPi);
        if (windPatternPhaseB > twoPi) windPatternPhaseB = std::fmod(windPatternPhaseB, twoPi);

        const double oscAX = std::sin(windPatternPhaseA);
        const double oscBX = std::sin(windPatternPhaseB + 1.61803398875);
        const double oscAY = std::cos(windPatternPhaseA * 0.79 + 0.35);
        const double oscBY = std::cos(windPatternPhaseB * 1.17 - 0.48);

        const double patternAmp = baseW * (0.05 + 0.55 * normDist);
        const double patternTargetX = (oscAX + 0.58 * oscBX) * patternAmp;
        const double patternTargetY = (oscAY + 0.58 * oscBY) * patternAmp;
        const double patternBlend = 0.12 + 0.20 * normDist;
        windPatternX = windPatternX * (1.0 - patternBlend) + patternTargetX * patternBlend;
        windPatternY = windPatternY * (1.0 - patternBlend) + patternTargetY * patternBlend;

        windNoiseX = windNoiseX * 0.72 + noiseDist(windRng) * noiseAmp * 0.28;
        windNoiseY = windNoiseY * 0.72 + noiseDist(windRng) * noiseAmp * 0.28;

        const double windForceX = windNoiseX + windPatternX * 0.42;
        const double windForceY = windNoiseY + windPatternY * 0.42;

        const double drag = 0.82 + (1.0 - normDist) * 0.10;
        windVelX = windVelX * drag + pullX + windForceX;
        windVelY = windVelY * drag + pullY + windForceY;

        const double vCap = std::max(0.65, baseM * (0.30 + 0.70 * normDist));
        const double newVelMag = std::hypot(windVelX, windVelY);
        if (newVelMag > vCap)
        {
            const double clip = vCap * clipDist(windRng);
            windVelX = (windVelX / newVelMag) * clip;
            windVelY = (windVelY / newVelMag) * clip;
        }

        windFracX += windVelX;
        windFracY += windVelY;

        int stepX = static_cast<int>(std::round(windFracX));
        int stepY = static_cast<int>(std::round(windFracY));
        if (stepX == 0 && stepY == 0)
            continue;

        windFracX -= static_cast<double>(stepX);
        windFracY -= static_cast<double>(stepY);
        windCarryX -= static_cast<double>(stepX);
        windCarryY -= static_cast<double>(stepY);
        queueMove(stepX, stepY);
    }

    const double carryCap = 120.0;
    const double finalCarryMag = std::hypot(windCarryX, windCarryY);
    if (finalCarryMag > carryCap)
    {
        const double s = carryCap / finalCarryMag;
        windCarryX *= s;
        windCarryY *= s;
    }
}

void MouseThread::resetWindState()
{
    constexpr double twoPi = 6.28318530717958647692;
    std::uniform_real_distribution<double> phaseDist(0.0, twoPi);
    std::uniform_real_distribution<double> rateDist(0.04, 0.16);

    windCarryX = 0.0;
    windCarryY = 0.0;
    windVelX = 0.0;
    windVelY = 0.0;
    windNoiseX = 0.0;
    windNoiseY = 0.0;
    windFracX = 0.0;
    windFracY = 0.0;
    windPatternX = 0.0;
    windPatternY = 0.0;
    windPatternPhaseA = phaseDist(windRng);
    windPatternPhaseB = phaseDist(windRng);
    windPatternRateA = rateDist(windRng);
    windPatternRateB = rateDist(windRng);
}

void MouseThread::appendWindDebugStep(int dx, int dy)
{
    if (dx == 0 && dy == 0)
        return;

    double deltaPxX = static_cast<double>(dx);
    double deltaPxY = static_cast<double>(dy);

    {
        std::lock_guard<std::mutex> cfgLock(configMutex);
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

        if (gpPtr && gpPtr->sens != 0.0 && gpPtr->yaw != 0.0 && gpPtr->pitch != 0.0)
        {
            const double fovNow = std::max(1.0, fov_x);
            const double fovScale = (gpPtr->fovScaled && gpPtr->baseFOV > 1.0) ? (fovNow / gpPtr->baseFOV) : 1.0;
            const double degX = static_cast<double>(dx) * gpPtr->sens * gpPtr->yaw * fovScale;
            const double degY = static_cast<double>(dy) * gpPtr->sens * gpPtr->pitch * fovScale;

            const double degPerPxX = fov_x / std::max(1.0, screen_width);
            const double degPerPxY = fov_y / std::max(1.0, screen_height);

            if (std::abs(degPerPxX) > 1e-8 && std::abs(degPerPxY) > 1e-8)
            {
                deltaPxX = degX / degPerPxX;
                deltaPxY = degY / degPerPxY;
            }
        }
    }

    std::lock_guard<std::mutex> lock(windDebugTrailMutex);
    const auto now = std::chrono::steady_clock::now();
    pruneWindDebugTrailLocked(now);

    if (windDebugTrail.empty())
    {
        windDebugCursorX = center_x;
        windDebugCursorY = center_y;
        windDebugTrail.push_back({ windDebugCursorX, windDebugCursorY, now });
    }

    windDebugCursorX += deltaPxX;
    windDebugCursorY += deltaPxY;
    windDebugTrail.push_back({ windDebugCursorX, windDebugCursorY, now });

    constexpr size_t maxTrailPoints = 220;
    while (windDebugTrail.size() > maxTrailPoints)
        windDebugTrail.pop_front();
}

void MouseThread::pruneWindDebugTrailLocked(const std::chrono::steady_clock::time_point& now)
{
    constexpr auto windTrailLifetime = std::chrono::milliseconds(900);
    while (!windDebugTrail.empty() && (now - windDebugTrail.front().t) > windTrailLifetime)
        windDebugTrail.pop_front();
}

std::pair<double, double> MouseThread::predict_target_position(double target_x, double target_y)
{
    auto current_time = std::chrono::steady_clock::now();

    if (prev_time.time_since_epoch().count() == 0 || !target_detected.load())
    {
        prev_time = current_time;
        prev_x = target_x;
        prev_y = target_y;
        prev_velocity_x = 0.0;
        prev_velocity_y = 0.0;
        return { target_x, target_y };
    }

    double dt = std::chrono::duration<double>(current_time - prev_time).count();
    if (dt < 1e-8) dt = 1e-8;

    double vx = (target_x - prev_x) / dt;
    double vy = (target_y - prev_y) / dt;

    vx = std::clamp(vx, -20000.0, 20000.0);
    vy = std::clamp(vy, -20000.0, 20000.0);

    prev_time = current_time;
    prev_x = target_x;
    prev_y = target_y;
    prev_velocity_x = vx;
    prev_velocity_y = vy;

    double predictedX = target_x + vx * prediction_interval;
    double predictedY = target_y + vy * prediction_interval;

    double detectionDelay = 0.05;
    if (config.backend == "DML")
    {
        detectionDelay = dml_detector->lastInferenceTimeDML.count();
    }
#ifdef USE_CUDA
    else
    {
        detectionDelay = trt_detector.lastInferenceTime.count();
    }
#endif
    predictedX += vx * detectionDelay;
    predictedY += vy * detectionDelay;

    return { predictedX, predictedY };
}

void MouseThread::sendMovementToDriver(int dx, int dy)
{
    if (dx == 0 && dy == 0)
    {
        return;
    }

    std::lock_guard<std::mutex> lock(input_method_mutex);

    if (kmbox_net)
    {
        kmbox_net->move(dx, dy);
    }
    else if (kmbox_a)
    {
        kmbox_a->move(dx, dy);
    }
    else if (makcu)
    {
        makcu->move(dx, dy);
    }
    else if (arduino)
    {
        arduino->move(dx, dy);
    }
    else if (gHub)
    {
        gHub->mouse_xy(dx, dy);
    }
    else
    {
        INPUT in{ 0 };
        in.type = INPUT_MOUSE;
        in.mi.dx = dx;  in.mi.dy = dy;
        in.mi.dwFlags = MOUSEEVENTF_MOVE | MOUSEEVENTF_VIRTUALDESK;
        SendInput(1, &in, sizeof(INPUT));
    }
}

std::pair<double, double> MouseThread::calc_movement(double tx, double ty)
{
    double offx = tx - center_x;
    double offy = ty - center_y;
    double dist = std::hypot(offx, offy);
    double speed = calculate_speed_multiplier(dist);

    double degPerPxX = fov_x / screen_width;
    double degPerPxY = fov_y / screen_height;

    double mmx = offx * degPerPxX;
    double mmy = offy * degPerPxY;

    double corr = 1.0;
    double fps = static_cast<double>(captureFps.load());
    if (fps > 30.0) corr = 30.0 / fps;

    auto counts_pair = config.degToCounts(mmx, mmy, fov_x);
    double move_x = counts_pair.first * speed * corr;
    double move_y = counts_pair.second * speed * corr;

    return { move_x, move_y };
}

double MouseThread::calculate_speed_multiplier(double distance)
{
    if (distance < config.snapRadius)
        return min_speed_multiplier * config.snapBoostFactor;

    if (distance < config.nearRadius)
    {
        double t = distance / config.nearRadius;
        double curve = 1.0 - std::pow(1.0 - t, config.speedCurveExponent);
        return min_speed_multiplier +
            (max_speed_multiplier - min_speed_multiplier) * curve;
    }

    double norm = std::clamp(distance / max_distance, 0.0, 1.0);
    return min_speed_multiplier +
        (max_speed_multiplier - min_speed_multiplier) * norm;
}

bool MouseThread::check_target_in_scope(double target_x, double target_y, double target_w, double target_h, double reduction_factor)
{
    double center_target_x = target_x + target_w / 2.0;
    double center_target_y = target_y + target_h / 2.0;

    double reduced_w = target_w * (reduction_factor / 2.0);
    double reduced_h = target_h * (reduction_factor / 2.0);

    double x1 = center_target_x - reduced_w;
    double x2 = center_target_x + reduced_w;
    double y1 = center_target_y - reduced_h;
    double y2 = center_target_y + reduced_h;

    return (center_x > x1 && center_x < x2 && center_y > y1 && center_y < y2);
}

void MouseThread::moveMouse(const AimbotTarget& target)
{
    std::lock_guard lg(input_method_mutex);

    auto predicted = predict_target_position(
        target.x + target.w / 2.0,
        target.y + target.h / 2.0);

    auto mv = calc_movement(predicted.first, predicted.second);
    queueMove(static_cast<int>(mv.first), static_cast<int>(mv.second));
}

void MouseThread::moveMousePivot(double pivotX, double pivotY)
{
    std::lock_guard lg(input_method_mutex);

    auto current_time = std::chrono::steady_clock::now();

    if (prev_time.time_since_epoch().count() == 0 || !target_detected.load())
    {
        prev_time = current_time;
        prev_x = pivotX; prev_y = pivotY;
        prev_velocity_x = prev_velocity_y = 0.0;

        auto m0 = calc_movement(pivotX, pivotY);
        const int mx0 = static_cast<int>(m0.first);
        const int my0 = static_cast<int>(m0.second);
        if (wind_mouse_enabled)
            windMouseMoveRelative(mx0, my0);
        else
            queueMove(mx0, my0);
        return;
    }

    double dt = std::chrono::duration<double>(current_time - prev_time).count();
    prev_time = current_time;
    dt = std::max(dt, 1e-8);

    double vx = std::clamp((pivotX - prev_x) / dt, -20000.0, 20000.0);
    double vy = std::clamp((pivotY - prev_y) / dt, -20000.0, 20000.0);
    prev_x = pivotX; prev_y = pivotY;
    prev_velocity_x = vx;  prev_velocity_y = vy;

    double predX = pivotX + vx * prediction_interval + vx * 0.002;
    double predY = pivotY + vy * prediction_interval + vy * 0.002;

    auto mv = calc_movement(predX, predY);
    int mx = static_cast<int>(mv.first);
    int my = static_cast<int>(mv.second);

    if (mx == 0 && my == 0)
    {
        return;
    }

    if (wind_mouse_enabled)
    {
        windMouseMoveRelative(mx, my);
    }
    else
    {
        queueMove(mx, my);
    }
}

void MouseThread::clearQueuedMoves()
{
    std::lock_guard<std::mutex> lock(queueMtx);
    std::queue<Move> empty;
    moveQueue.swap(empty);
    resetWindState();
}

void MouseThread::pressMouse(const AimbotTarget& target)
{
    std::lock_guard<std::mutex> lock(input_method_mutex);

    bool bScope = check_target_in_scope(target.x, target.y, target.w, target.h, bScope_multiplier);
    if (bScope && !mouse_pressed)
    {
        if (kmbox_net)
        {
            kmbox_net->leftDown();
        }
        else if (kmbox_a)
        {
            kmbox_a->leftDown();
        }
        else if (makcu)
        {
            makcu->press(0);
        }
        else if (arduino)
        {
            arduino->press();
        }
        else if (gHub)
        {
            gHub->mouse_down();
        }
        else
        {
            INPUT input = { 0 };
            input.type = INPUT_MOUSE;
            input.mi.dwFlags = MOUSEEVENTF_LEFTDOWN;
            SendInput(1, &input, sizeof(INPUT));
        }
        mouse_pressed = true;
    }
    else if (!bScope && mouse_pressed)
    {
        if (kmbox_net)
        {
            kmbox_net->leftUp();
        }
        else if (kmbox_a)
        {
            kmbox_a->leftUp();
        }
        else if (makcu)
        {
            makcu->release(0);
        }
        else if (arduino)
        {
            arduino->release();
        }
        else if (gHub)
        {
            gHub->mouse_up();
        }
        else
        {
            INPUT input = { 0 };
            input.type = INPUT_MOUSE;
            input.mi.dwFlags = MOUSEEVENTF_LEFTUP;
            SendInput(1, &input, sizeof(INPUT));
        }
        mouse_pressed = false;
    }
}

void MouseThread::releaseMouse()
{
    std::lock_guard<std::mutex> lock(input_method_mutex);

    if (mouse_pressed)
    {
        if (kmbox_net)
        {
            kmbox_net->leftUp();
        }
        else if (kmbox_a)
        {
            kmbox_a->leftUp();
        }
        else if (makcu)
        {
            makcu->release(0);
        }
        else if (arduino)
        {
            arduino->release();
        }
        else if (gHub)
        {
            gHub->mouse_up();
        }
        else
        {
            INPUT input = { 0 };
            input.type = INPUT_MOUSE;
            input.mi.dwFlags = MOUSEEVENTF_LEFTUP;
            SendInput(1, &input, sizeof(INPUT));
        }
        mouse_pressed = false;
    }
}

void MouseThread::resetPrediction()
{
    clearQueuedMoves();
    prev_time = std::chrono::steady_clock::time_point();
    prev_x = 0;
    prev_y = 0;
    prev_velocity_x = 0;
    prev_velocity_y = 0;
    target_detected.store(false);
}

void MouseThread::checkAndResetPredictions()
{
    auto current_time = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(current_time - last_target_time).count();

    if (elapsed > 0.5 && target_detected.load())
    {
        resetPrediction();
    }
}

std::vector<std::pair<double, double>> MouseThread::predictFuturePositions(double pivotX, double pivotY, int frames)
{
    std::vector<std::pair<double, double>> result;
    result.reserve(frames);

    const double fixedFps = 30.0;
    double frame_time = 1.0 / fixedFps;

    auto current_time = std::chrono::steady_clock::now();
    double dt = std::chrono::duration<double>(current_time - prev_time).count();

    if (prev_time.time_since_epoch().count() == 0 || dt > 0.5)
    {
        return result;
    }

    double vx = prev_velocity_x;
    double vy = prev_velocity_y;
    
    for (int i = 1; i <= frames; i++)
    {
        double t = frame_time * i;

        double px = pivotX + vx * t;
        double py = pivotY + vy * t;

        result.push_back({ px, py });
    }

    return result;
}

void MouseThread::storeFuturePositions(const std::vector<std::pair<double, double>>& positions)
{
    std::lock_guard<std::mutex> lock(futurePositionsMutex);
    futurePositions = positions;
}

void MouseThread::clearFuturePositions()
{
    std::lock_guard<std::mutex> lock(futurePositionsMutex);
    futurePositions.clear();
}

std::vector<std::pair<double, double>> MouseThread::getFuturePositions()
{
    std::lock_guard<std::mutex> lock(futurePositionsMutex);
    return futurePositions;
}

void MouseThread::clearWindDebugTrail()
{
    std::lock_guard<std::mutex> lock(windDebugTrailMutex);
    windDebugTrail.clear();
    windDebugCursorX = center_x;
    windDebugCursorY = center_y;
}

std::vector<std::pair<double, double>> MouseThread::getWindDebugTrail()
{
    std::lock_guard<std::mutex> lock(windDebugTrailMutex);
    const auto now = std::chrono::steady_clock::now();
    pruneWindDebugTrailLocked(now);

    std::vector<std::pair<double, double>> out;
    out.reserve(windDebugTrail.size());
    for (const auto& p : windDebugTrail)
        out.emplace_back(p.x, p.y);
    return out;
}

void MouseThread::setArduinoConnection(Arduino* newArduino)
{
    std::lock_guard<std::mutex> lock(input_method_mutex);
    arduino = newArduino;
}

void MouseThread::setKmboxAConnection(KmboxAConnection* newKmbox_a)
{
    std::lock_guard<std::mutex> lock(input_method_mutex);
    kmbox_a = newKmbox_a;
}

void MouseThread::setKmboxNetConnection(KmboxNetConnection* newKmbox_net)
{
    std::lock_guard<std::mutex> lock(input_method_mutex);
    kmbox_net = newKmbox_net;
}

void MouseThread::setMakcuConnection(MakcuConnection* newMakcu)
{
    std::lock_guard<std::mutex> lock(input_method_mutex);
    makcu = newMakcu;
}

void MouseThread::setGHubMouse(GhubMouse* newGHub)
{
    std::lock_guard<std::mutex> lock(input_method_mutex);
    gHub = newGHub;
}
