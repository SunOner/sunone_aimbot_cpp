#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <chrono>
#include <optional>
#include <vector>

#include "capture.h"
#include "mouse.h"
#include "sunone_aimbot_2.h"
#include "ghub.h"

extern GhubMouse* gHub;
#include "runtime/thread_loops.h"

void createInputDevices();
void assignInputDevices();
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


