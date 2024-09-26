#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <atomic>
#include <chrono>
#include <thread>
#include <iostream>

#include "keyboard_listener.h"
#include "mouse.h"
#include "keycodes.h"
#include "sunone_aimbot_cpp.h"

extern std::atomic<bool> shouldExit;
extern std::atomic<bool> aiming;
extern std::atomic<bool> detectionPaused;

extern MouseThread* globalMouseThread;

void keyboardListener()
{
    while (!shouldExit)
    {
        if (GetAsyncKeyState(KeyCodes::getKeyCode(config.button_targeting)) & 0x8000)
        {
            aiming = true;
        }
        else
        {
            aiming = false;
        }

        if (GetAsyncKeyState(KeyCodes::getKeyCode(config.button_exit)) & 0x8000)
        {
            shouldExit = true;
            quick_exit(0);
        }

        if (GetAsyncKeyState(KeyCodes::getKeyCode(config.button_pause)) & 0x8000)
        {
            detectionPaused = !detectionPaused;
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
        }

        if (GetAsyncKeyState(KeyCodes::getKeyCode(config.button_reload_config)) & 0x8000)
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