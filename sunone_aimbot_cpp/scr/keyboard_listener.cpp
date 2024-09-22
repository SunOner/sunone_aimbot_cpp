#include <atomic>
#include <chrono>
#include <thread>
#include <iostream>

#include "keyboard_listener.h"
#include "config.h"
#include "mouse.h"

extern std::atomic<bool> shouldExit;
extern std::atomic<bool> aiming;
extern std::atomic<bool> detectionPaused;
extern Config config;

extern MouseThread* globalMouseThread;

void keyboardListener()
{
    while (!shouldExit)
    {
        if (GetAsyncKeyState(VK_RBUTTON) & 0x8000)
        {
            aiming = true;
        }
        else
        {
            aiming = false;
        }

        if (GetAsyncKeyState(VK_F2) & 0x8000)
        {
            shouldExit = true;
            quick_exit(0);
        }

        if (GetAsyncKeyState(VK_F3) & 0x8000)
        {
            detectionPaused = !detectionPaused;
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
        }

        if (GetAsyncKeyState(VK_F4) & 0x8000)
        {
            config.loadConfig("config.ini");

            if (globalMouseThread)
            {
                globalMouseThread->updateConfig(
                    config.detection_resolution,

                    config.dpi,
                    config.sensitivity,
                    config.fovX,
                    config.fovY,
                    config.minSpeedMultiplier,
                    config.maxSpeedMultiplier,
                    config.predictionInterval,

                    config.auto_shoot,
                    config.bScope_multiplier
                );
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(300));
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}