#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <chrono>
#include <optional>
#include <vector>
#include <random>

#include "capture.h"
#include "mouse.h"
#include "sunone_aimbot_2.h"
#include "ghub.h"

extern GhubMouse* gHub;
extern Arduino* arduinoSerial;
extern KmboxNetConnection* kmboxNetSerial;
extern KmboxAConnection* kmboxASerial;
extern MakcuConnection* makcuSerial;

#include "runtime/thread_loops.h"

void createInputDevices();
void assignInputDevices();

std::chrono::steady_clock::time_point curve_start_time;
std::chrono::steady_clock::time_point shoot_start_time;
bool was_shooting = false;

static double total_moved_x = 0.0;
static float bezier_p1 = 0.0f;
static float bezier_p2 = 0.0f;
static double bezier_duration_sec = 2.0;
static float remainder_x = 0.0f;
static float remainder_y = 0.0f;

float calculate_cubic_bezier_pos(float t, float p1, float p2)
{
    float u = 1.0f - t;
    float term1 = 3.0f * u * u * t * p1;
    float term2 = 3.0f * u * t * t * p2;
    return term1 + term2;
}

void generate_new_curve()
{
    curve_start_time = std::chrono::steady_clock::now();
    total_moved_x = 0.0;
    static std::random_device rd;
    static std::mt19937 gen(rd());
    float effective_width = 20.0f + (config.recoil_sway * 1.5f);
    std::uniform_real_distribution<float> dist(-effective_width, effective_width);
    bezier_p1 = dist(gen);
    bezier_p2 = dist(gen);
    if (std::abs(bezier_p1) < 5.0f) bezier_p1 = (bezier_p1 < 0) ? -5.0f : 5.0f;
    if (std::abs(bezier_p2) < 5.0f) bezier_p2 = (bezier_p2 < 0) ? -5.0f : 5.0f;
    float actual_speed_decimal = config.recoil_sway_speed * 0.0001f;
    float speed_val = std::max(actual_speed_decimal, 0.00001f);
    bezier_duration_sec = 1.0f / (speed_val * 600.0f);
    if (bezier_duration_sec < 0.5) bezier_duration_sec = 0.5;
    if (bezier_duration_sec > 3.0) bezier_duration_sec = 3.0;
}

void handleEasyNoRecoil(MouseThread& mouseThread)
{
    bool currently_shooting = shooting.load() && zooming.load();
    if (!currently_shooting)
    {
        was_shooting = false;
        total_moved_x = 0.0;
        remainder_x = 0.0f;
        remainder_y = 0.0f;
        return;
    }
    
    auto now = std::chrono::steady_clock::now();
    if (currently_shooting && !was_shooting)
    {
        shoot_start_time = now;
        was_shooting = true;
        generate_new_curve();
    }
    
    double total_elapsed_ms = std::chrono::duration<double, std::milli>(now - shoot_start_time).count();
    double curve_elapsed_sec = std::chrono::duration<double>(now - curve_start_time).count();
    
    if (curve_elapsed_sec > bezier_duration_sec)
    {
        generate_new_curve();
        curve_elapsed_sec = 0.0;
    }
    
    if (config.easynorecoil)
    {
        std::lock_guard<std::mutex> lock(mouseThread.input_method_mutex);
        
        float move_y_float = config.recoil_pull_down_strength;
        float standard_x_float = 0.0f;
        
        if (total_elapsed_ms >= static_cast<double>(config.recoil_pull_left_delay))
            standard_x_float -= config.recoil_pull_left_strength;
            
        if (total_elapsed_ms >= static_cast<double>(config.recoil_pull_right_delay))
            standard_x_float += config.recoil_pull_right_strength;
            
        float bezier_delta_x = 0.0f;
        float t = static_cast<float>(curve_elapsed_sec / bezier_duration_sec);
        if (t > 1.0f) t = 1.0f;
        
        float target_pos_x = calculate_cubic_bezier_pos(t, bezier_p1, bezier_p2);
        bezier_delta_x = target_pos_x - static_cast<float>(total_moved_x);
        total_moved_x += bezier_delta_x;
        
        float total_move_x = standard_x_float + bezier_delta_x + remainder_x;
        float total_move_y = move_y_float + remainder_y;
        
        int final_x = static_cast<int>(total_move_x);
        int final_y = static_cast<int>(total_move_y);
        
        remainder_x = total_move_x - final_x;
        remainder_y = total_move_y - final_y;
        
        if (final_x != 0 || final_y != 0)
        {
            if (arduinoSerial) arduinoSerial->move(final_x, final_y);
            else if (gHub) gHub->mouse_xy(final_x, final_y);
            else if (kmboxNetSerial) kmboxNetSerial->move(final_x, final_y);
            else if (kmboxASerial) kmboxASerial->move(final_x, final_y);
            else if (makcuSerial) makcuSerial->move(final_x, final_y);
            else 
            {
                INPUT input = { 0 };
                input.type = INPUT_MOUSE;
                input.mi.dx = final_x;
                input.mi.dy = final_y;
                input.mi.dwFlags = MOUSEEVENTF_MOVE;
                SendInput(1, &input, sizeof(INPUT));
            }
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
