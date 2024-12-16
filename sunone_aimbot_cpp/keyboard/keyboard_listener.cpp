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
#include "capture.h"

extern std::atomic<bool> shouldExit;
extern std::atomic<bool> aiming;
extern std::atomic<bool> detectionPaused;

extern MouseThread* globalMouseThread;

bool isAnyKeyPressed(const std::vector<std::string>& keys)
{
    for (const auto& key_name : keys)
    {
        int key_code = KeyCodes::getKeyCode(key_name);
        if (key_code != -1 && (GetAsyncKeyState(key_code) & 0x8000))
        {
            return true;
        }
    }
    return false;
}

void keyboardListener()
{
    while (!shouldExit)
    {
        // Aiming
        if (!config.auto_aim)
        {
        aiming = isAnyKeyPressed(config.button_targeting) ||
            (config.arduino_enable_keys && serial && serial->isOpen() && serial->aiming_active);
        }
        else
        {
            aiming = true;
        }

        // Exit
        if (isAnyKeyPressed(config.button_exit))
        {
            shouldExit = true;
            quick_exit(0);
        }

        // Pause detection
        static bool pausePressed = false;
        if (isAnyKeyPressed(config.button_pause))
        {
            if (!pausePressed)
            {
                detectionPaused = !detectionPaused;
                pausePressed = true;
            }
        }
        else
        {
            pausePressed = false;
        }

        // Reload config
        static bool reloadPressed = false;
        if (isAnyKeyPressed(config.button_reload_config))
        {
            if (!reloadPressed)
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
                reloadPressed = true;
            }
        }
        else
        {
            reloadPressed = false;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}