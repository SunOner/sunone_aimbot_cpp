#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <atomic>
#include <chrono>
#include <thread>
#include <iostream>

#include "config.h"
#include "SerialConnection.h"
#include "keyboard_listener.h"
#include "mouse.h"
#include "keycodes.h"
#include "sunone_aimbot_cpp.h"
#include "capture.h"

extern std::atomic<bool> shouldExit;
extern std::atomic<bool> aiming;
extern std::atomic<bool> shooting;
extern std::atomic<bool> zooming;
extern std::atomic<bool> detectionPaused;

extern MouseThread* globalMouseThread;

// Offset adjustment global variables
std::atomic<bool> upArrowPressed(false);
std::atomic<bool> downArrowPressed(false);
std::atomic<bool> leftArrowPressed(false);
std::atomic<bool> rightArrowPressed(false);
const float OFFSET_STEP = 0.01f;
const float RECOIL_STEP = 10.0f;

// Arrow key vectors
const std::vector<std::string> upArrowKeys = { "UpArrow" };
const std::vector<std::string> downArrowKeys = { "DownArrow" };
const std::vector<std::string> leftArrowKeys = { "LeftArrow" };
const std::vector<std::string> rightArrowKeys = { "RightArrow" };
const std::vector<std::string> shiftKeys = { "LeftShift", "RightShift" };

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
            (config.arduino_enable_keys && arduinoSerial && arduinoSerial->isOpen() && arduinoSerial->aiming_active);
        }
        else
        {
            aiming = true;
        }

        // Shooting
        shooting = isAnyKeyPressed(config.button_shoot) ||
            (config.arduino_enable_keys && arduinoSerial && arduinoSerial->isOpen() && arduinoSerial->shooting_active);

        // Zooming
        zooming = isAnyKeyPressed(config.button_zoom) ||
            (config.arduino_enable_keys && arduinoSerial && arduinoSerial->isOpen() && arduinoSerial->zooming_active);

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
                config.loadConfig();
                
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

        // Arrow key detection logic using isAnyKeyPressed
        bool upArrow = isAnyKeyPressed(upArrowKeys);
        bool downArrow = isAnyKeyPressed(downArrowKeys);
        bool leftArrow = isAnyKeyPressed(leftArrowKeys);
        bool rightArrow = isAnyKeyPressed(rightArrowKeys);
        bool shiftKey = isAnyKeyPressed(shiftKeys);

        // Adjust offsets based on arrow keys and shift combination
        if (upArrow) {
            if (!upArrowPressed) { // Only process when key is first detected
                if (shiftKey) {
                    // Shift + Up Arrow: Decrease head offset
                    config.head_y_offset = std::max(0.0f, config.head_y_offset - OFFSET_STEP);
                } else {
                    // Up Arrow: Decrease body offset
                    config.body_y_offset = std::max(0.0f, config.body_y_offset - OFFSET_STEP);
                }
                upArrowPressed = true;
            }
        } else {
            upArrowPressed = false;
        }

        if (downArrow) {
            if (!downArrowPressed) { // Only process when key is first detected
                if (shiftKey) {
                    // Shift + Down Arrow: Increase head offset
                    config.head_y_offset = std::min(1.0f, config.head_y_offset + OFFSET_STEP);
                } else {
                    // Down Arrow: Increase body offset
                    config.body_y_offset = std::min(1.0f, config.body_y_offset + OFFSET_STEP);
                }
                downArrowPressed = true;
            }
        } else {
            downArrowPressed = false;
        }

        // Adjust easynorecoil strength using left/right arrow keys
        if (leftArrow) {
            if (!leftArrowPressed) { // Only process when key is first detected
                // Left Arrow: Decrease easynorecoil strength by 10
                config.easynorecoilstrength = std::max(0.0f, config.easynorecoilstrength - RECOIL_STEP);
                // Enable easynorecoil if adjusting strength
                if (!config.easynorecoil && config.easynorecoilstrength > 0.0f) {
                    config.easynorecoil = true;
                }
                leftArrowPressed = true;
            }
        } else {
            leftArrowPressed = false;
        }

        if (rightArrow) {
            if (!rightArrowPressed) { // Only process when key is first detected
                // Right Arrow: Increase easynorecoil strength by 10
                config.easynorecoilstrength = std::min(100.0f, config.easynorecoilstrength + RECOIL_STEP);
                // Enable easynorecoil if adjusting strength
                if (!config.easynorecoil && config.easynorecoilstrength > 0.0f) {
                    config.easynorecoil = true;
                }
                rightArrowPressed = true;
            }
        } else {
            rightArrowPressed = false;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}